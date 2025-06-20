import sys
import os
import threading
import logging
import time
import json
import sqlite3
import datetime
import re
import calendar
import shutil
from PyQt6.QtGui import QColor, QPalette
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLineEdit, QTableWidget, QLabel, QStatusBar,
    QMessageBox, QPushButton, QHBoxLayout, QHeaderView, QTableWidgetItem,
    QMenu, QSizePolicy, QDialog, QSpinBox, QCheckBox,
    QFileDialog, QComboBox, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QDesktopServices


# --- 日志配置 ---
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = time.strftime("FileScan_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger("FileScanApp")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("日志系统初始化完成。所有DEBUG级别日志将写入文件。")
    return logger


logger = setup_logging()

# --- 数据库管理类 ---
DB_NAME = "FileScan.db"


class DBManager:
    def __init__(self):
        logger.debug("DBManager: 初始化开始。")
        self.conn = None
        self.connect()
        self.create_table()
        self.create_indexes()
        logger.debug("DBManager: 初始化完成。")

    def connect(self):
        logger.debug(f"DBManager: 尝试连接数据库: {DB_NAME}")
        for attempt in range(3):
            try:
                self.conn = sqlite3.connect(DB_NAME, check_same_thread=False)
                logger.info(f"成功连接到数据库: {DB_NAME}")
                return
            except sqlite3.Error as e:
                logger.error(f"连接数据库失败 (尝试 {attempt + 1}/3): {e}")
                time.sleep(1)
        QMessageBox.critical(None, "数据库错误", "无法连接到数据库，程序将退出。")
        logger.critical("DBManager: 无法连接到数据库，程序即将退出。")
        sys.exit(1)

    def create_table(self):
        logger.debug("DBManager: 尝试创建/检查数据库表 'files'。")
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                path_lower TEXT PRIMARY KEY,
                original_path TEXT NOT NULL,
                ctime REAL NOT NULL,
                mtime REAL NOT NULL,
                atime REAL NOT NULL,
                extension TEXT,
                file_type TEXT,
                size REAL,
                is_dir INTEGER NOT NULL
            )
        ''')
        self.conn.commit()
        logger.info("数据库表 'files' 创建/检查完成。")
        logger.debug("DBManager: 数据库表结构检查通过。")

    def create_indexes(self):
        logger.debug("DBManager: 尝试创建/检查数据库索引。")
        cursor = self.conn.cursor()
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_type ON files(file_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_extension ON files(extension)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON files(ctime)')
        self.conn.commit()
        logger.info("数据库索引创建/检查完成。")
        logger.debug("DBManager: 数据库索引检查通过。")

    def insert_files_batch(self, file_list):
        logger.debug(f"DBManager: 批量插入 {len(file_list)} 个文件。")
        cursor = self.conn.cursor()
        try:
            cursor.executemany('''
                INSERT OR REPLACE INTO files (path_lower, original_path, ctime, mtime, atime, extension, file_type, size, is_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', file_list)
            self.conn.commit()
            logger.debug(f"DBManager: 成功批量插入/更新 {len(file_list)} 个文件。")
        except sqlite3.Error as e:
            logger.error(f"批量插入文件失败: {e}")
            logger.debug(f"DBManager: 批量插入文件失败，错误详情: {e}", exc_info=True)

    def delete_files_batch(self, path_lower_list):
        logger.debug(f"DBManager: 批量删除 {len(path_lower_list)} 个文件。")
        cursor = self.conn.cursor()
        try:
            cursor.executemany("DELETE FROM files WHERE path_lower = ?", [(pl,) for pl in path_lower_list])
            self.conn.commit()
            logger.debug(f"DBManager: 成功批量删除 {len(path_lower_list)} 个文件。")
        except sqlite3.Error as e:
            logger.error(f"批量删除文件失败: {e}")
            logger.debug(f"DBManager: 批量删除文件失败，错误详情: {e}", exc_info=True)

    def get_all_files(self):
        logger.debug("DBManager: 获取所有文件。")
        cursor = self.conn.cursor()
        cursor.execute("SELECT path_lower, original_path, mtime, is_dir FROM files")
        results = {}
        for row in cursor.fetchall():
            path_lower, original_path, mtime, is_dir = row
            results[path_lower] = {
                'original_path': original_path,
                'mtime': mtime,
                'is_dir': bool(is_dir)
            }
        logger.debug(f"DBManager: 检索到 {len(results)} 个文件项。")
        return results

    def search_files(self, query_lower, search_type="contains", limit=50000, sort_column="ctime", sort_order="DESC",
                     ctime_start=None, ctime_end=None):
        cursor = self.conn.cursor()
        conditions = []
        params = []

        if query_lower:
            if search_type == "contains":
                conditions.append("path_lower LIKE ?")
                params.append(f"%{query_lower}%")
            elif search_type == "starts_with":
                conditions.append("path_lower LIKE ?")
                params.append(f"{query_lower}%")
            elif search_type == "ends_with":
                conditions.append("path_lower LIKE ?")
                params.append(f"%{query_lower}")
            else:
                logger.warning(f"未知搜索类型: {search_type}，默认为 'contains'")
                conditions.append("path_lower LIKE ?")
                params.append(f"%{query_lower}%")

        if ctime_start is not None and ctime_end is not None:
            conditions.append("(ctime >= ? AND ctime < ?)")
            params.extend([ctime_start, ctime_end])

        sql_query = f"SELECT original_path, ctime, mtime, atime, extension, file_type, size, is_dir FROM files"

        if conditions:
            where_clause = " OR ".join(conditions)
            sql_query += f" WHERE {where_clause}"

        sql_query += f" ORDER BY {sort_column} {sort_order}"

        if limit:
            sql_query += f" LIMIT ?"
            params.append(limit)

        start_time = time.time()
        logger.debug(f"DBManager: 执行SQL查询: {sql_query} 参数: {params}")
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        end_time = time.time()
        logger.debug(
            f"DBManager: 数据库搜索 '{query_lower}' (日期范围: {ctime_start}-{ctime_end}) 找到 {len(results)} 个结果，耗时 {(end_time - start_time):.4f} 秒。")
        return results

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭。")
            logger.debug("DBManager: 数据库连接关闭完成。")


# --- 配置管理类 ---
CONFIG_FILE = "config.json"


class ConfigManager:
    def __init__(self):
        logger.debug("ConfigManager: 初始化开始。")
        self.config = self._load_config()
        logger.debug("ConfigManager: 初始化完成。")

    def _load_config(self):
        logger.debug(f"ConfigManager: 尝试从 '{CONFIG_FILE}' 加载配置。")
        default_config = {
            "indexed_paths": [],
            "path_colors": {},
            "path_cleanup_enabled": {},
            "cleanup_days": 30,
            "size_warning_mb": 1000,
            "search_delay_ms": 200,
            "max_display_results": 50000,
            "column_visibility": {
                "name": True, "path": True, "type": True, "ctime": True, "mtime": True,
                "atime": True, "extension": True, "file_type": True, "size": True
            },
            "theme": "light"
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                for key, value in default_config.items():
                    config.setdefault(key, value)
                logger.info(f"成功从 '{CONFIG_FILE}' 加载配置。")
                logger.debug(f"ConfigManager: 加载的配置: {config}")
                return config
            except json.JSONDecodeError as e:
                logger.error(f"加载配置文件 '{CONFIG_FILE}' 失败 (JSON 格式错误): {e}", exc_info=True)
                QMessageBox.warning(None, "配置加载错误", f"无法读取配置文件，可能是文件损坏。将使用默认配置。\n错误: {e}")
            except Exception as e:
                logger.error(f"加载配置文件 '{CONFIG_FILE}' 失败: {e}", exc_info=True)
                QMessageBox.warning(None, "配置加载错误", f"无法加载配置文件。将使用默认配置。\n错误: {e}")
        logger.info("未找到配置文件或加载失败，将使用默认配置。")
        logger.debug(f"ConfigManager: 使用默认配置: {default_config}")
        return default_config

    def save_config(self):
        logger.debug(f"ConfigManager: 尝试保存配置到 '{CONFIG_FILE}'。")
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"成功将配置保存到 '{CONFIG_FILE}'。")
            logger.debug(f"ConfigManager: 保存的配置: {self.config}")
        except Exception as e:
            logger.error(f"保存配置文件 '{CONFIG_FILE}' 失败: {e}", exc_info=True)
            QMessageBox.warning(None, "配置保存错误", f"无法保存配置文件，请检查权限或磁盘空间。\n错误: {e}")

    def get(self, key, default=None):
        value = self.config.get(key, default)
        logger.debug(f"ConfigManager: 获取配置项 '{key}': {value}")
        return value

    def set(self, key, value):
        self.config[key] = value
        logger.debug(f"ConfigManager: 设置配置项 '{key}' 为: {value}")


# --- 后台索引线程 ---
def get_file_type(extension):
    extension = extension.lower().lstrip('.')
    type_map = {
        'doc': '文档', 'docx': '文档', 'pdf': '文档', 'txt': '文本', 'rtf': '文档',
        'odt': '文档', 'md': '文本',
        'jpg': '图片', 'jpeg': '图片', 'png': '图片', 'gif': '图片', 'bmp': '图片',
        'tiff': '图片', 'svg': '图片',
        'exe': '应用程序', 'bat': '应用程序', 'sh': '应用程序', 'app': '应用程序',
        'log': '日志',
    }
    return type_map.get(extension, '其他')


class IndexerThread(QThread):
    indexing_started = pyqtSignal()
    indexing_progress = pyqtSignal(str)
    indexing_finished = pyqtSignal(int)
    indexing_error = pyqtSignal(str)

    def __init__(self, db_manager, paths_to_index, last_index_times):
        super().__init__()
        self.db_manager = db_manager
        self.paths_to_index = paths_to_index
        self.last_index_times = last_index_times
        self.stop_event = threading.Event()
        logger.info(f"IndexerThread: 初始化，将扫描路径: {self.paths_to_index}")
        logger.debug(f"IndexerThread: 传入的上次索引时间: {self.last_index_times}")

    def run(self):
        logger.info("IndexerThread: 索引构建线程开始运行。")
        self.indexing_started.emit()

        existing_db_index = self.db_manager.get_all_files()
        logger.info(f"IndexerThread: 从数据库加载 {len(existing_db_index)} 个现有索引项。")

        current_scan_paths = set()
        processed_count = 0
        batch_size = 1000
        batch_files = []

        start_time = time.time()

        for path in self.paths_to_index:
            if self.stop_event.is_set():
                logger.info("IndexerThread: 索引线程收到停止信号，正在退出。")
                break

            if not os.path.exists(path):
                logger.warning(f"IndexerThread: 扫描路径 '{path}' 不存在或不可访问，跳过。")
                self.indexing_progress.emit(f"跳过不存在的路径: {path}")
                continue

            logger.info(f"IndexerThread: 开始扫描路径: {path}")
            self.indexing_progress.emit(f"正在扫描: {path}")
            try:
                for root, dirs, files in os.walk(path, topdown=True):
                    if self.stop_event.is_set():
                        logger.info(f"IndexerThread: 路径 {path} 扫描中收到停止信号。")
                        break

                    dirs[:] = [d for d in dirs if
                               not d.startswith(('.', '$')) and d not in ['System Volume Information', '$RECYCLE.BIN']]
                    logger.debug(f"IndexerThread: 扫描目录: {root}, 文件数: {len(files)}, 子目录数: {len(dirs)}")

                    for name in files + dirs:
                        full_path = os.path.join(root, name).replace('\\', '/')

                        path_lower = full_path.lower()
                        current_scan_paths.add(path_lower)
                        path_lower = full_path.lower()
                        current_scan_paths.add(path_lower)

                        try:
                            stats = os.stat(full_path)
                            mtime = stats.st_mtime

                            if (path_lower in existing_db_index and
                                    existing_db_index[path_lower]['mtime'] == mtime and
                                    existing_db_index[path_lower]['original_path'] == full_path):
                                logger.debug(f"IndexerThread: 跳过未修改的文件: {full_path}")
                                continue

                            ctime = stats.st_ctime
                            atime = stats.st_atime
                            is_dir = os.path.isdir(full_path)
                            extension = os.path.splitext(name)[1].lower() if not is_dir else ''
                            file_type = get_file_type(extension) if not is_dir else '文件夹'
                            size = stats.st_size if not is_dir else None

                            batch_files.append(
                                (path_lower, full_path, ctime, mtime, atime, extension, file_type, size, is_dir)
                            )
                            processed_count += 1
                            if len(batch_files) >= batch_size:
                                self.db_manager.insert_files_batch(batch_files)
                                batch_files = []
                                self.indexing_progress.emit(f"已索引 {processed_count} 个项 (当前: {root})")
                                logger.debug(f"IndexerThread: 达到批次大小，已批量插入。当前索引项: {processed_count}")

                        except OSError as e:
                            logger.warning(f"IndexerThread: 无法访问路径 {full_path}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"IndexerThread: 处理文件 {full_path} 时发生未知错误: {e}", exc_info=True)
                            continue
            except Exception as e:
                logger.error(f"IndexerThread: 扫描 '{path}' 时发生错误: {e}", exc_info=True)
                self.indexing_error.emit(f"扫描 '{path}' 失败: {e}")

            if self.stop_event.is_set():
                break

        if batch_files:
            self.db_manager.insert_files_batch(batch_files)
            self.indexing_progress.emit(f"已索引 {processed_count} 个项 (完成批量插入)")
            logger.debug(f"IndexerThread: 剩余批次文件已插入。")

        logger.info("IndexerThread: 开始检查并清理数据库中过时的条目...")
        deleted_count = 0
        batch_delete = []
        norm_indexed_paths = [os.path.normpath(p) for p in self.paths_to_index]
        logger.debug(f"IndexerThread: 标准化后的索引路径: {norm_indexed_paths}")

        for path_lower, data in existing_db_index.items():
            original_path = data['original_path']

            is_in_scope = False
            for p_idx in norm_indexed_paths:
                if os.path.commonpath([os.path.normpath(original_path), p_idx]) == p_idx:
                    is_in_scope = True
                    break

            if not is_in_scope or (path_lower not in current_scan_paths):
                batch_delete.append(path_lower)
                deleted_count += 1
                logger.debug(
                    f"IndexerThread: 标记删除: {original_path} (在范围外: {not is_in_scope}, 未找到: {path_lower not in current_scan_paths})")

            if len(batch_delete) >= batch_size:
                self.db_manager.delete_files_batch(batch_delete)
                self.indexing_progress.emit(f"已清理 {deleted_count} 个过时条目")
                logger.debug(f"IndexerThread: 达到批次大小，已批量删除过时条目。")
                batch_delete = []

        if batch_delete:
            self.db_manager.delete_files_batch(batch_delete)
            logger.debug(f"IndexerThread: 剩余批次过时条目已删除。")

        logger.info(f"IndexerThread: 过时条目清理完成，共删除 {deleted_count} 个条目。")

        total_indexed_count = len(self.db_manager.get_all_files())
        end_time = time.time()
        duration = end_time - start_time
        logger.info(
            f"IndexerThread: 索引构建/更新完成。数据库中共有 {total_indexed_count} 个文件/文件夹，本次耗时 {duration:.2f} 秒。")
        logger.info(f"IndexerThread: 本次操作新增/修改项约 {processed_count} 个，删除项约 {deleted_count} 个。")
        self.indexing_finished.emit(total_indexed_count)

    def stop(self):
        self.stop_event.set()
        logger.info("IndexerThread: 已发送停止信号给索引线程。")


# --- 后台搜索线程 ---
class SearchWorker(QThread):
    search_results_ready = pyqtSignal(list, str, float)

    def __init__(self, db_manager, query, search_type="contains", max_results=50000, sort_column="ctime",
                 sort_order="DESC"):
        super().__init__()
        self.db_manager = db_manager
        self.query = query
        self.query_lower = query.lower()
        self.search_type = search_type
        self.max_results = max_results
        self.sort_column = sort_column
        self.sort_order = sort_order
        self.stop_event = threading.Event()
        logger.debug(
            f"SearchWorker: 初始化，查询: '{self.query}', 类型: {self.search_type}, 排序: {self.sort_column} {self.sort_order}")

    def run(self):
        logger.info(f"SearchWorker: 搜索线程开始运行，查询: '{self.query}'")
        start_time = time.time()

        if self.stop_event.is_set():
            logger.info("SearchWorker: 搜索线程在开始前收到停止信号，正在退出。")
            self.search_results_ready.emit([], self.query, 0.0)
            return

        ctime_start, ctime_end = None, None
        date_match = re.fullmatch(r'(\d{4})(?:-|\s)?(\d{1,2})?(?:-|\s)?(\d{1,2})?', self.query.strip())

        if date_match:
            try:
                year = int(date_match.group(1))
                month = int(date_match.group(2)) if date_match.group(2) else None
                day = int(date_match.group(3)) if date_match.group(3) else None

                if day and month:
                    start_dt = datetime.datetime(year, month, day)
                    end_dt = start_dt + datetime.timedelta(days=1)
                    ctime_start = start_dt.timestamp()
                    ctime_end = end_dt.timestamp()
                    logger.info(f"SearchWorker: 检测到日期搜索模式 (日): {year}-{month}-{day}")
                elif month:
                    start_dt = datetime.datetime(year, month, 1)
                    _, num_days = calendar.monthrange(year, month)
                    end_dt = start_dt + datetime.timedelta(days=num_days)
                    ctime_start = start_dt.timestamp()
                    ctime_end = end_dt.timestamp()
                    logger.info(f"SearchWorker: 检测到日期搜索模式 (月): {year}-{month}")
                else:
                    start_dt = datetime.datetime(year, 1, 1)
                    end_dt = datetime.datetime(year + 1, 1, 1)
                    ctime_start = start_dt.timestamp()
                    ctime_end = end_dt.timestamp()
                    logger.info(f"SearchWorker: 检测到日期搜索模式 (年): {year}")

                self.query_lower = ""
                logger.debug("SearchWorker: 纯日期搜索，query_lower 已清空。")

            except ValueError as e:
                logger.warning(f"SearchWorker: '{self.query}' 看起来像日期，但解析失败: {e}。将作为常规文本搜索。")
                ctime_start, ctime_end = None, None
            except Exception as e:
                logger.error(f"SearchWorker: 日期解析过程中发生未知错误: {e}", exc_info=True)
                ctime_start, ctime_end = None, None

        results = self.db_manager.search_files(
            self.query_lower, self.search_type, self.max_results,
            self.sort_column, self.sort_order, ctime_start, ctime_end
        )

        end_time = time.time()
        duration = end_time - start_time

        if self.stop_event.is_set():
            logger.info("SearchWorker: 搜索线程收到停止信号（在结果生成后），正在退出。")
            self.search_results_ready.emit([], self.query, 0.0)
            return

        self.search_results_ready.emit(results, self.query, duration)
        logger.info(f"SearchWorker: 搜索完成。查询 '{self.query}' 找到 {len(results)} 个结果，耗时 {duration:.4f} 秒。")

    def stop(self):
        self.stop_event.set()
        logger.info("SearchWorker: 已发送停止信号给搜索线程。")


# --- 清理线程 (已修改) ---
class CleanupThread(QThread):
    cleanup_finished = pyqtSignal(int)
    cleanup_error = pyqtSignal(str)
    size_warning = pyqtSignal(str, float)
    cleanup_by_size_started = pyqtSignal(str)
    cleanup_by_size_finished = pyqtSignal(str, int, float)  # path, deleted_count, freed_mb

    # 新增：实时进度信号
    cleanup_size_progress = pyqtSignal(str, str)  # path, progress_message
    cleanup_time_progress = pyqtSignal(str, str)  # path, progress_message

    def __init__(self, db_manager, indexed_paths, path_cleanup_enabled, cleanup_days, size_warning_mb):
        super().__init__()
        self.db_manager = db_manager
        self.indexed_paths = indexed_paths
        self.path_cleanup_enabled = path_cleanup_enabled
        self.cleanup_days = cleanup_days
        self.size_warning_mb = size_warning_mb
        self.stop_event = threading.Event()
        logger.info("CleanupThread: 初始化。")
        logger.debug(f"CleanupThread: 清理天数: {self.cleanup_days}, 大小警告MB: {self.size_warning_mb}")

    def run(self):
        logger.info("CleanupThread: 清理线程开始运行。")
        # 新增：用于跟踪本轮已发出大小警告的路径集合
        paths_warned_this_overall_round = set()
        try:
            while not self.stop_event.is_set():
                if not self.indexed_paths:
                    logger.debug("CleanupThread: 无扫描路径，清理线程将每5分钟检查一次。")
                    self.stop_event.wait(300)
                    continue

                # 在每一轮新的整体检查开始时，清空已警告路径的集合
                paths_warned_this_overall_round.clear()
                logger.debug("CleanupThread: 新一轮整体检查开始，清空已警告路径记录。")

                # --- 1. 执行基于时间的定期清理 ---
                logger.debug("CleanupThread: 开始执行基于时间的清理。")
                deleted_count_timed_total = 0

                for path in self.indexed_paths:
                    if self.stop_event.is_set(): break
                    if not self.path_cleanup_enabled.get(path, False):
                        # logger.debug(f"CleanupThread (Time): 路径 {path} 未启用清理，跳过。") # 日志可以简化
                        continue
                    if not os.path.exists(path):
                        # logger.warning(f"CleanupThread (Time): 路径 {path} 不存在，跳过清理。") # 日志可以简化
                        continue
                    # ... (时间清理的内部逻辑保持不变，此处省略) ...
                    # 假设时间清理代码在这里
                    try:
                        current_time = time.time()
                        cutoff_time = current_time - self.cleanup_days * 86400
                        # logger.debug( # 简化日志
                        #     f"CleanupThread (Time): 路径 {path} 时间清理截止时间戳: {cutoff_time} "
                        #     f"(对应 {datetime.datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d %H:%M:%S')})"
                        # )

                        like_path_time = path.replace('\\', '/').lower()
                        if not like_path_time.endswith('/'):
                            like_path_time += '/'

                        cursor = self.db_manager.conn.cursor()
                        sql_query_time = "SELECT path_lower, original_path FROM files WHERE path_lower LIKE ? AND ctime < ? AND is_dir = 0"
                        params_time = (f"{like_path_time}%", cutoff_time)

                        cursor.execute(sql_query_time, params_time)
                        files_to_delete_time = cursor.fetchall()
                        # logger.debug(f"CleanupThread (Time): 路径 {path} 找到 {len(files_to_delete_time)} 个过期文件。") # 简化日志

                        total_to_delete_time_path = len(files_to_delete_time)
                        deleted_this_path_timed = 0
                        batch_delete_db_time = []

                        for i, (path_lower, original_path) in enumerate(files_to_delete_time):
                            if self.stop_event.is_set(): break
                            try:
                                if os.path.exists(original_path):
                                    os.remove(original_path)
                                    # logger.info(f"CleanupThread (Time): 成功删除文件: {original_path}") # 简化日志
                                    batch_delete_db_time.append(path_lower)
                                    deleted_this_path_timed += 1
                                else:
                                    # logger.debug(f"CleanupThread (Time): 文件 {original_path} 不存在，从数据库移除。") # 简化日志
                                    batch_delete_db_time.append(path_lower)

                                if deleted_this_path_timed > 0 and (
                                        deleted_this_path_timed % 10 == 0 or i == total_to_delete_time_path - 1):
                                    progress_msg = f"基于时间: 已删 {deleted_this_path_timed}/{total_to_delete_time_path} 过期项"
                                    self.cleanup_time_progress.emit(path, progress_msg)
                                    self.stop_event.wait(0.01)

                            except OSError as e:
                                logger.warning(f"CleanupThread (Time): 无法删除文件 {original_path}: {e}",
                                               exc_info=True)

                        if batch_delete_db_time:
                            self.db_manager.delete_files_batch(batch_delete_db_time)
                            deleted_count_timed_total += deleted_this_path_timed
                            # logger.debug(f"CleanupThread (Time): 清理路径 {path} 的数据库记录批次已提交。") # 简化日志

                    except Exception as e:
                        logger.error(f"CleanupThread (Time): 清理路径 {path} 时发生错误: {e}", exc_info=True)
                        self.cleanup_error.emit(f"时间清理路径 {path} 失败: {e}")

                if self.stop_event.is_set(): break

                if deleted_count_timed_total > 0:
                    self.cleanup_finished.emit(deleted_count_timed_total)
                    logger.info(
                        f"CleanupThread: 本次所有路径基于时间的清理操作共删除 {deleted_count_timed_total} 个文件。")

                # --- 2. 循环检查并执行基于大小的空间清理 ---
                logger.debug("CleanupThread: 开始循环检查文件夹大小并执行空间清理。")
                performed_size_cleanup_this_overall_round = False

                for path in self.indexed_paths:
                    if self.stop_event.is_set(): break
                    if not self.path_cleanup_enabled.get(path, False):
                        # logger.debug(f"CleanupThread (Size): 路径 {path} 未启用大小清理，跳过。") # 简化
                        continue

                    user_defined_threshold_bytes = self.size_warning_mb * 1024 * 1024
                    target_cleanup_size_bytes = user_defined_threshold_bytes * 0.80

                    total_deleted_for_path_size = 0
                    total_freed_bytes_for_path_size = 0

                    # 标记此路径是否在本轮首次超限时触发了警告
                    # 这个标记是针对单个路径在多次迭代中的，与外层的 overall_round 不同
                    warning_emitted_for_this_path_session = False

                    while not self.stop_event.is_set():  # 内循环，持续清理单个路径
                        if not os.path.exists(path):
                            logger.warning(f"CleanupThread (Size): 路径 {path} 在清理中消失，终止对该路径的清理。")
                            break

                        try:
                            current_actual_size_bytes = 0
                            # (os.walk 来获取 current_actual_size_bytes 的代码保持不变，此处省略)
                            for root_s, dirs_s, files_s in os.walk(path):
                                if self.stop_event.is_set(): break
                                for file_s in files_s:
                                    if self.stop_event.is_set(): break
                                    try:
                                        current_actual_size_bytes += os.path.getsize(os.path.join(root_s, file_s))
                                    except OSError:
                                        continue
                            if self.stop_event.is_set(): break

                            current_actual_size_mb = current_actual_size_bytes / (1024 * 1024)
                            target_cleanup_size_mb = target_cleanup_size_bytes / (1024 * 1024)

                            # logger.debug(f"CleanupThread (Size): 路径 '{os.path.basename(path)}' 当前大小: {current_actual_size_mb:.2f} MB. "
                            #              f"警告阈值: {self.size_warning_mb} MB. "
                            #              f"目标清理后大小 (阈值的80%): {target_cleanup_size_mb:.2f} MB.") # 可以简化或按需保留

                            if current_actual_size_bytes <= user_defined_threshold_bytes:
                                # logger.info(f"CleanupThread (Size): 路径 '{os.path.basename(path)}' "
                                #             f"大小 {current_actual_size_mb:.2f} MB 未超阈值 {self.size_warning_mb} MB。无需清理。") # 简化
                                break

                            if current_actual_size_bytes <= target_cleanup_size_bytes:
                                # logger.info(f"CleanupThread (Size): 路径 '{os.path.basename(path)}' "
                                #             f"大小 {current_actual_size_mb:.2f} MB 已达目标 {target_cleanup_size_mb:.2f} MB。清理完成。") # 简化
                                break

                                # ---- 修改点：控制警告弹窗只出现一次 ----
                            if not warning_emitted_for_this_path_session and path not in paths_warned_this_overall_round:
                                self.size_warning.emit(path, current_actual_size_mb)  # 触发弹窗警告
                                paths_warned_this_overall_round.add(path)  # 记录本轮已警告
                                warning_emitted_for_this_path_session = True  # 标记此路径的当前清理会话已警告
                                logger.info(
                                    f"CleanupThread (Size): 路径 '{os.path.basename(path)}' 首次检测超限，已发出警告。")

                            # 无论是否弹窗，只要超限且未达目标，就开始/继续清理
                            performed_size_cleanup_this_overall_round = True
                            self.cleanup_by_size_started.emit(path)  # 这个信号通知UI“正在清理XX路径”

                            # (获取最旧文件的SQL查询和删除逻辑保持不变，此处省略)
                            # ...
                            like_path_size = path.replace('\\', '/').lower()
                            if not like_path_size.endswith('/'):
                                like_path_size += '/'

                            cursor = self.db_manager.conn.cursor()
                            cursor.execute(
                                "SELECT path_lower, original_path, size FROM files WHERE path_lower LIKE ? AND is_dir = 0 ORDER BY ctime ASC LIMIT 1",
                                (f"{like_path_size}%",)
                            )
                            oldest_file_data = cursor.fetchone()

                            if not oldest_file_data:
                                logger.warning(
                                    f"CleanupThread (Size): 路径 '{os.path.basename(path)}' 超限但数据库中无文件可删。")
                                break

                            path_lower_s, original_path_s, _ = oldest_file_data

                            deleted_this_iteration = 0
                            freed_bytes_this_iteration = 0
                            db_paths_to_delete_s = []

                            try:
                                if os.path.exists(original_path_s):
                                    actual_deleted_file_size = os.path.getsize(original_path_s)
                                    os.remove(original_path_s)
                                    # logger.info(f"CleanupThread (Size): 已删除 '{original_path_s}' (释放: {actual_deleted_file_size / (1024*1024):.2f} MB).") # 简化
                                    db_paths_to_delete_s.append(path_lower_s)
                                    freed_bytes_this_iteration += actual_deleted_file_size
                                    deleted_this_iteration += 1
                                else:
                                    # logger.warning(f"CleanupThread (Size): 文件 '{original_path_s}' 在删除前已不存在，将从DB移除。") # 简化
                                    db_paths_to_delete_s.append(path_lower_s)
                            except OSError as e:
                                logger.warning(f"CleanupThread (Size): 无法删除文件 '{original_path_s}': {e}")

                            if db_paths_to_delete_s:
                                self.db_manager.delete_files_batch(db_paths_to_delete_s)

                            if deleted_this_iteration > 0:
                                total_deleted_for_path_size += deleted_this_iteration
                                total_freed_bytes_for_path_size += freed_bytes_this_iteration
                                freed_mb_iter = freed_bytes_this_iteration / (1024 * 1024)
                                # 状态栏进度信息
                                progress_msg_size = (
                                    f"基于大小 '{os.path.basename(path)}': 删 {deleted_this_iteration}项, "
                                    f"释放 {freed_mb_iter:.2f}MB. "
                                    f"累计为该路径删 {total_deleted_for_path_size}项, "
                                    f"共释放 {total_freed_bytes_for_path_size / (1024 * 1024):.2f}MB. "
                                    f"当前大小: {current_actual_size_mb - freed_mb_iter:.2f}MB / 目标 {target_cleanup_size_mb:.2f}MB")
                                self.cleanup_size_progress.emit(path, progress_msg_size)

                            self.stop_event.wait(0.2)  # 稍作等待，让UI有机会刷新，也给文件系统一点时间

                        except Exception as e:
                            logger.error(f"CleanupThread (Size): 清理路径 '{os.path.basename(path)}' 内部循环出错: {e}",
                                         exc_info=True)
                            self.cleanup_error.emit(f"大小清理路径 {path} 失败: {e}")
                            break

                    if total_deleted_for_path_size > 0:
                        total_freed_mb_for_path_size = total_freed_bytes_for_path_size / (1024 * 1024)
                        self.cleanup_by_size_finished.emit(path, total_deleted_for_path_size,
                                                           total_freed_mb_for_path_size)
                        logger.info(f"CleanupThread (Size): 路径 '{os.path.basename(path)}' 大小清理会话结束。 "
                                    f"总共删除 {total_deleted_for_path_size} 个文件, 释放 {total_freed_mb_for_path_size:.2f} MB。")

                if self.stop_event.is_set(): break

                if performed_size_cleanup_this_overall_round:
                    logger.debug(
                        "CleanupThread: 本轮至少为一个路径进行了大小清理，将在5秒后开始下一轮整体检查。")  # 缩短再次检查的间隔
                    self.stop_event.wait(5)
                else:
                    logger.debug("CleanupThread: 所有路径均未进行大小清理（或已达标），清理线程将休眠1小时。")
                    self.stop_event.wait(3600)

        except Exception as e:
            logger.critical(f"CleanupThread: 清理线程主循环发生未预期严重错误: {e}", exc_info=True)
            self.cleanup_error.emit(f"清理线程崩溃: {e}")
    def stop(self):
        self.stop_event.set()
        logger.info("CleanupThread: 已发送停止信号给清理线程。")


# --- 设置窗口 ---
class SettingsWindow(QDialog):
    settings_saved = pyqtSignal(dict)

    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setGeometry(200, 200, 600, 600)
        self.setMinimumSize(500, 500)
        self.config_manager = config_manager
        logger.debug("SettingsWindow: 初始化UI。")
        self._init_ui()
        logger.debug("SettingsWindow: 加载设置。")
        self._load_settings()
        logger.debug("SettingsWindow: 设置窗口初始化完成。")

    def _init_ui(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        path_group = QGroupBox("扫描路径")
        path_layout = QVBoxLayout()
        path_group.setLayout(path_layout)
        path_layout.addWidget(QLabel("管理扫描路径:"))
        path_list_layout = QHBoxLayout()
        self.path_table = QTableWidget()
        self.path_table.setColumnCount(3)
        self.path_table.setHorizontalHeaderLabels(["路径", "颜色", "启用清理"])
        self.path_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.path_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.path_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.path_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        path_list_layout.addWidget(self.path_table)
        path_buttons_layout = QVBoxLayout()
        self.add_path_button = QPushButton("添加路径...")
        self.add_path_button.clicked.connect(self._add_scan_path)
        path_buttons_layout.addWidget(self.add_path_button)
        self.remove_path_button = QPushButton("移除选中路径")
        self.remove_path_button.clicked.connect(self._remove_scan_path)
        path_buttons_layout.addWidget(self.remove_path_button)
        path_buttons_layout.addStretch(1)
        path_list_layout.addLayout(path_buttons_layout)
        path_layout.addLayout(path_list_layout)
        main_layout.addWidget(path_group)

        performance_group = QGroupBox("性能设置")
        performance_layout = QVBoxLayout()
        performance_group.setLayout(performance_layout)
        delay_layout = QHBoxLayout()
        self.delay_spinbox = QSpinBox()
        self.delay_spinbox.setRange(50, 9999)
        delay_layout.addWidget(QLabel("搜索延迟 (毫秒):"))
        delay_layout.addWidget(self.delay_spinbox)
        delay_layout.addStretch(1)
        performance_layout.addLayout(delay_layout)
        max_results_layout = QHBoxLayout()
        self.max_results_spinbox = QSpinBox()
        self.max_results_spinbox.setRange(1, 9999999)
        max_results_layout.addWidget(QLabel("最大显示结果数:"))
        max_results_layout.addWidget(self.max_results_spinbox)
        max_results_layout.addStretch(1)
        performance_layout.addLayout(max_results_layout)
        main_layout.addWidget(performance_group)

        cleanup_group = QGroupBox("清理设置")
        cleanup_layout = QVBoxLayout()
        cleanup_group.setLayout(cleanup_layout)
        cleanup_days_layout = QHBoxLayout()
        self.cleanup_days_spinbox = QSpinBox()
        self.cleanup_days_spinbox.setRange(1, 365)
        cleanup_days_layout.addWidget(QLabel("清理超过的天数:"))
        cleanup_days_layout.addWidget(self.cleanup_days_spinbox)
        cleanup_days_layout.addStretch(1)
        cleanup_layout.addLayout(cleanup_days_layout)
        size_warning_layout = QHBoxLayout()
        self.size_warning_spinbox = QSpinBox()
        self.size_warning_spinbox.setRange(100, 999999)
        size_warning_layout.addWidget(QLabel("文件夹大小警告阈值 (MB):"))
        size_warning_layout.addWidget(self.size_warning_spinbox)
        size_warning_layout.addStretch(1)
        cleanup_layout.addLayout(size_warning_layout)
        main_layout.addWidget(cleanup_group)

        display_group = QGroupBox("显示设置")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)
        theme_layout = QHBoxLayout()
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色", "深色"])
        theme_layout.addWidget(QLabel("主题:"))
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch(1)
        display_layout.addLayout(theme_layout)
        display_layout.addWidget(QLabel("选择显示的列:"))

        self.column_checkboxes = {}
        column_names = {"name": "文件名", "path": "路径", "type": "类型", "ctime": "创建时间", "mtime": "修改时间",
                        "atime": "访问时间", "extension": "扩展名", "file_type": "文件类型", "size": "大小"}

        columns_per_row = 4
        all_column_keys = list(column_names.keys())
        for i in range(0, len(all_column_keys), columns_per_row):
            row_layout = QHBoxLayout()
            for key in all_column_keys[i:i + columns_per_row]:
                checkbox = QCheckBox(column_names[key])
                self.column_checkboxes[key] = checkbox
                row_layout.addWidget(checkbox)
            row_layout.addStretch(1)
            display_layout.addLayout(row_layout)
        main_layout.addWidget(display_group)

        button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存")
        self.save_button.clicked.connect(self._save_settings)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch(1)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        logger.debug("SettingsWindow: UI元素创建完成。")

    def _load_settings(self):
        logger.debug("SettingsWindow: 开始加载设置到UI。")
        indexed_paths = self.config_manager.get("indexed_paths", [])
        path_colors = self.config_manager.get("path_colors", {})
        path_cleanup_enabled = self.config_manager.get("path_cleanup_enabled", {})
        self.path_table.setRowCount(len(indexed_paths))
        color_options = ["白色", "红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色"]

        for row, path in enumerate(indexed_paths):
            path_item = QTableWidgetItem(path)
            path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.path_table.setItem(row, 0, path_item)

            color_combo = QComboBox()
            color_combo.addItems(color_options)
            color_combo.setCurrentText(path_colors.get(path, "白色"))
            self.path_table.setCellWidget(row, 1, color_combo)

            cleanup_checkbox = QCheckBox()
            cleanup_checkbox.setChecked(path_cleanup_enabled.get(path, False))
            cleanup_widget = QWidget()
            cleanup_layout = QHBoxLayout(cleanup_widget)
            cleanup_layout.addWidget(cleanup_checkbox)
            cleanup_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cleanup_layout.setContentsMargins(0, 0, 0, 0)
            self.path_table.setCellWidget(row, 2, cleanup_widget)
            logger.debug(
                f"SettingsWindow: 加载路径: {path}, 颜色: {path_colors.get(path, '白色')}, 清理启用: {path_cleanup_enabled.get(path, False)}")

        self.delay_spinbox.setValue(self.config_manager.get("search_delay_ms", 200))
        self.max_results_spinbox.setValue(self.config_manager.get("max_display_results", 50000))
        self.cleanup_days_spinbox.setValue(self.config_manager.get("cleanup_days", 30))
        self.size_warning_spinbox.setValue(self.config_manager.get("size_warning_mb", 1000))
        column_visibility = self.config_manager.get("column_visibility", {})
        for key, checkbox in self.column_checkboxes.items():
            checkbox.setChecked(column_visibility.get(key, True))
            logger.debug(f"SettingsWindow: 列 '{key}' 可见性: {column_visibility.get(key, True)}")
        theme = self.config_manager.get("theme", "light")
        self.theme_combo.setCurrentText("深色" if theme == "dark" else "浅色")
        logger.debug(f"SettingsWindow: 加载主题: {theme}")
        logger.debug("SettingsWindow: 设置加载完成。")

    def _add_scan_path(self):
        logger.debug("SettingsWindow: 尝试添加扫描路径。")
        folder = QFileDialog.getExistingDirectory(self, "选择要扫描的文件夹")
        if folder:
            current_paths = [self.path_table.item(i, 0).text() for i in range(self.path_table.rowCount())]
            if folder not in current_paths:
                row = self.path_table.rowCount()
                self.path_table.insertRow(row)
                path_item = QTableWidgetItem(folder)
                path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.path_table.setItem(row, 0, path_item)
                color_combo = QComboBox()
                color_combo.addItems(["白色", "红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色"])
                self.path_table.setCellWidget(row, 1, color_combo)
                cleanup_checkbox = QCheckBox()
                cleanup_widget = QWidget()
                cleanup_layout = QHBoxLayout(cleanup_widget)
                cleanup_layout.addWidget(cleanup_checkbox)
                cleanup_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cleanup_layout.setContentsMargins(0, 0, 0, 0)
                self.path_table.setCellWidget(row, 2, cleanup_widget)
                logger.info(f"SettingsWindow: 添加新扫描路径: {folder}")
            else:
                QMessageBox.warning(self, "路径重复", "该路径已在列表中。")
                logger.warning(f"SettingsWindow: 尝试添加重复路径: {folder}")
        else:
            logger.debug("SettingsWindow: 用户取消添加路径。")

    def _remove_scan_path(self):
        logger.debug("SettingsWindow: 尝试移除选中路径。")
        selected_rows = sorted(list(set(index.row() for index in self.path_table.selectedIndexes())), reverse=True)
        if not selected_rows:
            QMessageBox.warning(self, "未选择", "请选择要移除的路径。")
            logger.warning("SettingsWindow: 未选择任何路径进行移除。")
            return
        reply = QMessageBox.question(self, "确认移除", "确定要移除选中的路径吗？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for row in selected_rows:
                path_to_remove = self.path_table.item(row, 0).text()
                self.path_table.removeRow(row)
                logger.info(f"SettingsWindow: 移除扫描路径: {path_to_remove}")
        else:
            logger.debug("SettingsWindow: 用户取消移除路径。")

    def _save_settings(self):
        logger.debug("SettingsWindow: 尝试保存设置。")
        indexed_paths = [self.path_table.item(i, 0).text() for i in range(self.path_table.rowCount())]
        path_colors = {self.path_table.item(i, 0).text(): self.path_table.cellWidget(i, 1).currentText() for i in
                       range(self.path_table.rowCount())}
        path_cleanup_enabled = {
            self.path_table.item(i, 0).text(): self.path_table.cellWidget(i, 2).findChild(QCheckBox).isChecked() for i
            in range(self.path_table.rowCount())}

        new_config = {
            "indexed_paths": indexed_paths,
            "path_colors": path_colors,
            "path_cleanup_enabled": path_cleanup_enabled,
            "cleanup_days": self.cleanup_days_spinbox.value(),
            "size_warning_mb": self.size_warning_spinbox.value(),
            "search_delay_ms": self.delay_spinbox.value(),
            "max_display_results": self.max_results_spinbox.value(),
            "column_visibility": {key: checkbox.isChecked() for key, checkbox in self.column_checkboxes.items()},
            "theme": "dark" if self.theme_combo.currentText() == "深色" else "light"
        }
        logger.debug(f"SettingsWindow: 准备保存的新配置: {new_config}")

        for key, value in new_config.items():
            self.config_manager.set(key, value)

        self.config_manager.save_config()
        self.settings_saved.emit(new_config)
        self.accept()
        logger.info("SettingsWindow: 设置保存完成并应用。")


# --- 主应用程序窗口 ---
class FileScanWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("File Scan")
        self.setGeometry(100, 100, 1200, 700)

        logger.info("FileScanWindow: 主窗口初始化开始。")
        self.db_manager = DBManager()
        self.config_manager = ConfigManager()

        self._load_config()

        self.last_index_times = {}
        self.sort_column = "ctime"
        self.sort_order = Qt.SortOrder.DescendingOrder
        self.results_cache = []

        self.current_search_worker = None
        self.indexer_thread = None
        self.cleanup_thread = None
        self.last_query = ""

        self.search_delay_timer = QTimer(self)
        self.search_delay_timer.setSingleShot(True)
        self.search_delay_timer.timeout.connect(self._start_search)

        self.auto_index_timer = QTimer(self)
        self.auto_index_timer.timeout.connect(self._trigger_auto_index)
        self.auto_index_interval_ms = 15 * 60 * 1000  # 15分钟

        self._init_ui()
        self._apply_theme()
        logger.debug("FileScanWindow: 首次启动索引。")
        self._start_indexing(self.indexed_paths, is_manual=True)
        QTimer.singleShot(5000, self._start_cleanup_thread)
        logger.info("FileScanWindow: 主窗口 UI 元素加载完成。")
        logger.debug("FileScanWindow: 主窗口初始化完成。")

    def _load_config(self):
        logger.debug("FileScanWindow: 加载应用程序配置。")
        self.indexed_paths = self.config_manager.get("indexed_paths", [])
        self.path_colors = self.config_manager.get("path_colors", {})
        self.path_cleanup_enabled = self.config_manager.get("path_cleanup_enabled", {})
        self.cleanup_days = self.config_manager.get("cleanup_days", 30)
        self.size_warning_mb = self.config_manager.get("size_warning_mb", 1000)
        self.typing_delay_ms = self.config_manager.get("search_delay_ms", 200)
        self.max_display_results = self.config_manager.get("max_display_results", 50000)
        self.column_visibility = self.config_manager.get("column_visibility", {
            "name": True, "path": True, "type": True, "ctime": True, "mtime": True,
            "atime": True, "extension": True, "file_type": True, "size": True
        })
        self.theme = self.config_manager.get("theme", "light")
        logger.debug(f"FileScanWindow: 配置加载完毕: indexed_paths={self.indexed_paths}, theme={self.theme}")

    def _apply_theme(self):
        logger.debug(f"FileScanWindow: 应用主题: {self.theme}")
        stylesheet = ""
        if self.theme == "dark":
            stylesheet = """
                QMainWindow, QDialog { background-color: #2b2b2b; color: #f0f0f0; }
                QLineEdit { border: 1px solid #555; border-radius: 5px; padding: 5px; background-color: #3c3f41; color: #f0f0f0; }
                QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 5px 10px; }
                QPushButton:hover { background-color: #005ba1; } QPushButton:pressed { background-color: #003f7a; }
                QTableWidget { border: 1px solid #555; background-color: #3c3f41; color: #f0f0f0; alternate-background-color: #4a4a4a; gridline-color: #666; }
                QTableWidget::item { padding: 3px; }
                QTableWidget::item:selected { background-color: #0078d7; color: white; }
                QHeaderView::section { background-color: #555; color: white; padding: 5px; border: none; border-bottom: 1px solid #666; }
                QStatusBar { background-color: #2b2b2b; color: white; }
                QProgressBar { border: 1px solid #555; border-radius: 5px; text-align: center; background-color: #3c3f41; color: white; }
                QProgressBar::chunk { background-color: #0078d7; }
                QGroupBox { border: 1px solid #555; border-radius: 5px; margin-top: 10px; color: #f0f0f0; }
                QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }
                QLabel { color: #f0f0f0; }
                QCheckBox { color: #f0f0f0; }
                QSpinBox { background-color: #3c3f41; color: #f0f0f0; border: 1px solid #555; padding: 2px; }
                QComboBox { background-color: #3c3f41; color: #f0f0f0; border: 1px solid #555; padding: 2px; }
            """
        else:  # light
            stylesheet = """
                QPushButton { background-color: #0078d7; color: white; border: none; border-radius: 5px; padding: 5px 10px; }
                QPushButton:hover { background-color: #005ba1; } QPushButton:pressed { background-color: #003f7a; }
                QTableWidget { gridline-color: #ccc; }
                QHeaderView::section { background-color: #e0e0e0; color: black; }
                QGroupBox { color: black; }
                QLabel { color: black; }
                QCheckBox { color: black; }
                QSpinBox, QComboBox { background-color: white; color: black; border: 1px solid #ccc; }
            """
        self.setStyleSheet(stylesheet)
        logger.debug("FileScanWindow: 主题应用完成。")

    def _init_ui(self):
        logger.debug("FileScanWindow: 初始化主UI布局和控件。")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("在这里输入文件名、路径或日期(YYYY-MM-DD)进行搜索...")
        self.search_input.setFixedHeight(30)
        self.search_input.textChanged.connect(self._on_search_input_changed)
        search_layout.addWidget(self.search_input)

        self.reindex_button = QPushButton("重新索引")
        self.reindex_button.clicked.connect(lambda: self._start_indexing(self.indexed_paths, is_manual=True))
        self.reindex_button.setToolTip("重新扫描所有文件并构建索引")
        search_layout.addWidget(self.reindex_button)

        self.settings_button = QPushButton("设置")
        self.settings_button.clicked.connect(self._open_settings)
        self.settings_button.setToolTip("配置扫描路径和搜索选项")
        search_layout.addWidget(self.settings_button)
        main_layout.addLayout(search_layout)

        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self._update_table_columns()
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.itemDoubleClicked.connect(self._open_selected_item)
        self.results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_context_menu)
        self.results_table.horizontalHeader().setSortIndicatorShown(True)
        self.results_table.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        main_layout.addWidget(self.results_table)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(0)
        main_layout.addWidget(self.progress_bar)

        self.status_bar = self.statusBar()
        self.index_count_label = QLabel("索引文件: 0")
        self.result_count_label = QLabel("结果: 0")
        self.status_bar.addPermanentWidget(self.index_count_label)
        self.status_bar.addPermanentWidget(self.result_count_label)
        self.status_bar.showMessage("准备就绪。")
        logger.debug("FileScanWindow: 主UI初始化完成。")

    def _update_table_columns(self):
        logger.debug("FileScanWindow: 更新表格列显示。")
        self.visible_columns = [
            ("name", "文件名"), ("path", "路径"), ("type", "类型"), ("ctime", "创建时间"),
            ("mtime", "修改时间"), ("atime", "访问时间"), ("extension", "扩展名"),
            ("file_type", "文件类型"), ("size", "大小")
        ]
        self.visible_columns = [col for col in self.visible_columns if self.column_visibility.get(col[0], True)]
        self.results_table.setColumnCount(len(self.visible_columns))
        self.results_table.setHorizontalHeaderLabels([col[1] for col in self.visible_columns])

        header = self.results_table.horizontalHeader()
        for i in range(len(self.visible_columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)

        try:
            sort_col_keys = [col[0] for col in self.visible_columns]
            reverse_map = {"original_path": ["name", "path"], "is_dir": ["type"]}
            ui_col_to_find = self.sort_column
            for ui_name, db_names in reverse_map.items():
                if self.sort_column in db_names:
                    ui_col_to_find = ui_name
                    break

            if ui_col_to_find in sort_col_keys:
                sort_col_index = sort_col_keys.index(ui_col_to_find)
                self.results_table.horizontalHeader().setSortIndicator(sort_col_index, self.sort_order)
                logger.debug(
                    f"FileScanWindow: 恢复排序指示器到列 {self.sort_column} (UI: {ui_col_to_find}), 顺序 {self.sort_order}")
            else:
                self.sort_column = "ctime"
                self.sort_order = Qt.SortOrder.DescendingOrder
                if "ctime" in sort_col_keys:
                    ctime_index = sort_col_keys.index("ctime")
                    self.results_table.horizontalHeader().setSortIndicator(ctime_index, self.sort_order)
                    logger.debug("FileScanWindow: 原排序列被隐藏，重置排序为 ctime。")
                else:
                    self.results_table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
                    logger.debug("FileScanWindow: 无法找到 ctime 列，隐藏排序指示器。")
        except Exception as e:
            logger.error(f"FileScanWindow: 更新表格列时恢复排序指示器失败: {e}", exc_info=True)
            self.results_table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)

    def _on_header_clicked(self, logical_index):
        logger.debug(f"FileScanWindow: 表头点击事件，逻辑索引: {logical_index}")
        if logical_index < 0 or logical_index >= len(self.visible_columns):
            logger.warning(f"FileScanWindow: 无效的表头点击逻辑索引: {logical_index}")
            return

        column_key = self.visible_columns[logical_index][0]
        sort_key_map = {
            "name": "original_path",
            "path": "original_path",
            "type": "is_dir"
        }
        db_sort_column = sort_key_map.get(column_key, column_key)
        if self.sort_column == db_sort_column:
            self.sort_order = Qt.SortOrder.AscendingOrder if self.sort_order == Qt.SortOrder.DescendingOrder else Qt.SortOrder.DescendingOrder
        else:
            self.sort_column = db_sort_column
            self.sort_order = Qt.SortOrder.AscendingOrder

        logger.debug(f"FileScanWindow: 新排序设置: 列 '{self.sort_column}', 顺序 '{self.sort_order}'")
        self.results_table.horizontalHeader().setSortIndicator(logical_index, self.sort_order)
        self._start_search()

    def _open_settings(self):
        logger.debug("FileScanWindow: 打开设置窗口。")
        settings_dialog = SettingsWindow(self.config_manager, self)
        settings_dialog.settings_saved.connect(self._apply_settings)
        settings_dialog.exec()

    def _apply_settings(self, new_settings):
        logger.info("FileScanWindow: 应用新设置。")
        old_indexed_paths = set(self.indexed_paths)
        self._load_config()
        new_indexed_paths = set(self.indexed_paths)

        self.search_delay_timer.setInterval(self.typing_delay_ms)
        self._apply_theme()
        self._update_table_columns()
        logger.info(
            f"FileScanWindow: 应用新设置: 搜索延迟 {self.typing_delay_ms}ms, 最大结果 {self.max_display_results}")

        logger.debug("FileScanWindow: 重启清理线程以应用新设置。")
        self._start_cleanup_thread()

        if old_indexed_paths != new_indexed_paths:
            logger.info("FileScanWindow: 扫描路径已更改，将触发重新索引。")
            self._start_indexing(self.indexed_paths, is_manual=True)
        else:
            logger.debug("FileScanWindow: 扫描路径未更改，仅刷新搜索结果。")
            self._start_search()

    def _trigger_auto_index(self):
        logger.info("FileScanWindow: 自动索引计时器触发。")
        self._start_indexing(self.indexed_paths, is_manual=False)

    def _start_indexing(self, paths_to_index, is_manual):
        logger.debug(
            f"FileScanWindow: 尝试启动索引 (手动: {is_manual})。当前索引器状态: {self.indexer_thread and self.indexer_thread.isRunning()}")
        if self.indexer_thread and self.indexer_thread.isRunning():
            if is_manual:
                QMessageBox.information(self, "索引正在进行", "一个索引任务已在后台运行，请稍后再试。")
            logger.warning(f"FileScanWindow: 索引触发 (手动: {is_manual})，但已有任务在运行，本次跳过。")
            return

        if not paths_to_index:
            if is_manual:
                QMessageBox.warning(self, "扫描路径未设置", "请在'设置'中添加需要扫描的文件夹。")
            self.status_bar.showMessage("请设置扫描路径。", 5000)
            logger.warning("FileScanWindow: 无扫描路径，无法启动索引。")
            return

        self.auto_index_timer.stop()
        logger.debug("FileScanWindow: 自动索引计时器已暂停。")

        if is_manual:
            self.status_bar.showMessage("正在构建/更新文件索引，请稍候...")
            self.results_table.setRowCount(0)
            self.search_input.setEnabled(False)
            self.reindex_button.setEnabled(False)
            self.settings_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            logger.debug("FileScanWindow: 进入手动索引UI锁定状态。")
        else:
            self.status_bar.showMessage("正在后台自动更新索引...", 5000)
            logger.debug("FileScanWindow: 进入自动索引后台状态。")

        self.indexer_thread = IndexerThread(self.db_manager, paths_to_index, self.last_index_times)
        self.indexer_thread.indexing_started.connect(self._on_indexing_started)
        self.indexer_thread.indexing_progress.connect(self._on_indexing_progress)
        self.indexer_thread.indexing_finished.connect(lambda count: self._on_indexing_finished(count, is_manual))
        self.indexer_thread.indexing_error.connect(lambda msg: self._on_indexing_error(msg, is_manual))
        self.indexer_thread.start()
        logger.info(f"FileScanWindow: 已启动文件索引线程 (手动: {is_manual})")

    def _on_indexing_started(self):
        logger.info("FileScanWindow: 文件索引开始信号接收。")

    def _on_indexing_progress(self, message):
        self.status_bar.showMessage(f"索引进度: {message}")
        logger.debug(f"FileScanWindow: 索引进度更新: {message}")

    def _on_indexing_finished(self, total_count, is_manual):
        self.auto_index_timer.start(self.auto_index_interval_ms)
        logger.info(
            f"FileScanWindow: 索引完成。自动扫描计时器已在 {self.auto_index_interval_ms / (60 * 1000):.1f} 分钟后启动/重置。")

        if is_manual:
            self.search_input.setEnabled(True)
            self.reindex_button.setEnabled(True)
            self.settings_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            logger.debug("FileScanWindow: 退出手动索引UI锁定状态。")

        self.index_count_label.setText(f"索引文件: {total_count}")
        self.status_bar.showMessage(f"索引完成，数据库共 {total_count} 个项。", 8000)
        logger.info(f"FileScanWindow: 索引完成，总计 {total_count} 个项。")

        for path in self.indexed_paths:
            if os.path.exists(path):
                try:
                    self.last_index_times[path] = os.stat(path).st_mtime
                    logger.debug(f"FileScanWindow: 更新路径 '{path}' 的上次索引时间。")
                except OSError as e:
                    logger.warning(f"FileScanWindow: 无法获取路径 '{path}' 的修改时间: {e}")

        self._start_search()

    def _on_indexing_error(self, message, is_manual):
        self.auto_index_timer.start(self.auto_index_interval_ms)
        logger.error(f"FileScanWindow: 索引过程中发生错误: {message}")
        if is_manual:
            self.search_input.setEnabled(True)
            self.reindex_button.setEnabled(True)
            self.settings_button.setEnabled(True)
            self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"索引错误: {message}", 5000)

    def _start_cleanup_thread(self):
        logger.debug("FileScanWindow: 尝试启动清理线程。")
        if self.cleanup_thread and self.cleanup_thread.isRunning():
            logger.info("FileScanWindow: 停止现有清理线程。")
            self.cleanup_thread.stop()
            self.cleanup_thread.wait(5000)
            if self.cleanup_thread.isRunning():
                logger.warning("FileScanWindow: 清理线程未能按时停止。")

        if not self.indexed_paths:
            logger.info("FileScanWindow: 无扫描路径，跳过启动清理线程。")
            return

        self.cleanup_thread = CleanupThread(
            self.db_manager, self.indexed_paths, self.path_cleanup_enabled,
            self.cleanup_days, self.size_warning_mb
        )
        self.cleanup_thread.cleanup_finished.connect(self._on_cleanup_finished)
        self.cleanup_thread.cleanup_error.connect(self._on_cleanup_error)
        self.cleanup_thread.size_warning.connect(self._on_size_warning)
        self.cleanup_thread.cleanup_by_size_started.connect(self._on_cleanup_by_size_started)
        self.cleanup_thread.cleanup_by_size_finished.connect(self._on_cleanup_by_size_finished)

        # 新增：连接实时进度信号
        self.cleanup_thread.cleanup_size_progress.connect(self._on_cleanup_size_progress)
        self.cleanup_thread.cleanup_time_progress.connect(self._on_cleanup_time_progress)

        self.cleanup_thread.start()
        logger.info("FileScanWindow: 清理线程已启动。")

    def _on_cleanup_finished(self, deleted_count):
        # 仅在有文件被删除时显示总结信息，避免覆盖掉实时进度
        if deleted_count > 0:
            self.status_bar.showMessage(f"自动按时间清理完成，共删除 {deleted_count} 个过期文件。", 5000)
            logger.info(f"FileScanWindow: 基于时间的清理完成，删除 {deleted_count} 个文件。")
            self._start_search()
        else:
            logger.info("FileScanWindow: 基于时间的清理运行完成，没有文件被删除。")

    def _on_cleanup_error(self, message):
        self.status_bar.showMessage(f"清理错误: {message}", 5000)
        logger.error(f"FileScanWindow: 清理线程报告错误: {message}")

    def _on_size_warning(self, path, size_mb):
        logger.warning(f"FileScanWindow: 收到文件夹大小警告: 路径='{path}', 大小='{size_mb:.2f} MB'")
        message = f"路径 {path} 的大小为 {size_mb:.2f} MB，超过设置的阈值 {self.size_warning_mb} MB。"
        if self.path_cleanup_enabled.get(path, False):
            message += "\n\n由于已启用清理，将开始自动删除最旧的文件以释放空间。"
        QMessageBox.warning(self, "文件夹大小警告", message)

    def _on_cleanup_by_size_started(self, path):
        message = f"路径 '{path}' 超出大小限制，正在后台自动清理..."
        self.status_bar.showMessage(message, 5000)
        logger.info(f"FileScanWindow: {message}")

    def _on_cleanup_by_size_finished(self, path, deleted_count, freed_mb):
        # 仅在有文件被删除时显示总结信息
        if deleted_count > 0:
            message = f"空间清理完成: 为路径 '{path}' 删除了 {deleted_count} 个文件，释放了 {freed_mb:.2f} MB。"
            self.status_bar.showMessage(message, 8000)
            logger.info(f"FileScanWindow: {message}")
            self._start_search()
        else:
            logger.info(f"FileScanWindow: 空间清理运行完成，但没有文件被删除。")
            self.status_bar.showMessage(f"路径 '{path}' 空间清理已尝试，无文件可删。", 5000)

    # --- 新增：处理实时清理进度的槽函数 ---
    def _on_cleanup_size_progress(self, path, message):
        """处理基于大小的清理进度"""
        self.status_bar.showMessage(f"空间清理 '{os.path.basename(path)}': {message}")

    def _on_cleanup_time_progress(self, path, message):
        """处理基于时间的清理进度"""
        self.status_bar.showMessage(f"时间清理 '{os.path.basename(path)}': {message}")

    def _on_search_input_changed(self, text):
        logger.debug(f"FileScanWindow: 搜索输入框文本改变: '{text}'。重置搜索延迟计时器。")
        self.search_delay_timer.stop()
        self.search_delay_timer.start(self.typing_delay_ms)

    def _start_search(self):
        query = self.search_input.text().strip()
        logger.debug(f"FileScanWindow: 启动搜索，查询: '{query}'。")

        if self.current_search_worker and self.current_search_worker.isRunning():
            logger.debug("FileScanWindow: 停止上一个搜索工作线程。")
            self.current_search_worker.stop()
            self.current_search_worker.wait(1000)

        self.last_query = query
        self.status_bar.showMessage(f"正在搜索 '{query}'...")
        if not self.indexer_thread or not self.indexer_thread.isRunning():
            self.progress_bar.setVisible(True)
            logger.debug("FileScanWindow: 索引器未运行，显示搜索进度条。")

        sort_order_str = "ASC" if self.sort_order == Qt.SortOrder.AscendingOrder else "DESC"
        logger.debug(
            f"FileScanWindow: 创建新的搜索工作线程: query='{query}', sort_column='{self.sort_column}', sort_order='{sort_order_str}'")
        self.current_search_worker = SearchWorker(
            self.db_manager, query, max_results=self.max_display_results,
            sort_column=self.sort_column, sort_order=sort_order_str
        )
        self.current_search_worker.search_results_ready.connect(self._on_search_results_ready)
        self.current_search_worker.start()

    def _on_search_results_ready(self, results, query_str, duration):
        logger.debug(
            f"FileScanWindow: 收到搜索结果，查询: '{query_str}', 结果数: {len(results)}, 耗时: {duration:.3f}s")
        if query_str != self.last_query:
            logger.debug(f"FileScanWindow: 忽略过时的搜索结果。当前查询: '{self.last_query}'")
            return

        self.progress_bar.setVisible(False)
        self.results_table.setSortingEnabled(False)
        logger.debug("FileScanWindow: 禁用表格排序以更新结果。")

        try:
            self.results_table.setRowCount(0)
            self.results_cache = results
            display_count = len(results)
            self.result_count_label.setText(f"结果: {display_count}")

            if display_count > self.max_display_results:
                self.status_bar.showMessage(
                    f"找到 {display_count} 个结果，显示前 {self.max_display_results} 个。用时 {duration:.3f} 秒。")
                display_count = self.max_display_results
                logger.info(f"FileScanWindow: 结果过多，只显示前 {self.max_display_results} 个。")
            else:
                self.status_bar.showMessage(f"找到 {display_count} 个结果，用时 {duration:.3f} 秒。")

            if not results:
                self.results_table.setRowCount(1)
                msg = "无结果" if query_str else "数据库为空，请在设置中添加扫描路径并重新索引。"
                self.results_table.setItem(0, 0, QTableWidgetItem(msg))
                logger.info(f"FileScanWindow: 搜索结果为空。显示信息: '{msg}'")
                self.results_table.setSortingEnabled(True)
                return

            self.results_table.setRowCount(display_count)
            column_indices = {col[0]: idx for idx, col in enumerate(self.visible_columns)}
            color_map = {"白色": "", "红色": "#ffdddd", "橙色": "#ffeedd", "黄色": "#ffffdd", "绿色": "#ddffdd",
                         "青色": "#ddffff", "蓝色": "#ddddff", "紫色": "#eeddff", "粉色": "#ffddf0"}

            path_color_mapping = {}
            for indexed_path, color_name in self.path_colors.items():
                if color_name != "白色":
                    path_color_mapping[os.path.normpath(indexed_path).lower()] = color_map.get(color_name)

            for row_idx, (path, ctime, mtime, atime, extension, file_type, size, is_dir) in enumerate(
                    results[:display_count]):
                file_name = os.path.basename(path)
                directory = os.path.dirname(path)

                size_str = ''
                if size is not None and not is_dir:
                    if size > 1024 * 1024 * 1024:
                        size_str = f"{size / (1024 ** 3):.2f} GB"
                    elif size > 1024 * 1024:
                        size_str = f"{size / (1024 ** 2):.2f} MB"
                    else:
                        size_str = f"{size / 1024:.2f} KB"

                row_data = {
                    "name": file_name, "path": directory,
                    "type": "文件夹" if is_dir else "文件",
                    "ctime": datetime.datetime.fromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S') if ctime else '',
                    "mtime": datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S') if mtime else '',
                    "atime": datetime.datetime.fromtimestamp(atime).strftime('%Y-%m-%d %H:%M:%S') if atime else '',
                    "extension": extension, "file_type": file_type, "size": size_str
                }

                row_background_color = None
                normalized_path = os.path.normpath(path).lower()
                for indexed_path_lower, color_hex in path_color_mapping.items():
                    if normalized_path.startswith(indexed_path_lower):
                        row_background_color = color_hex
                        break

                for key, val in row_data.items():
                    if key in column_indices:
                        item = QTableWidgetItem(val)
                        item.setData(Qt.ItemDataRole.UserRole, path)

                        if row_background_color:
                            item.setBackground(QColor(row_background_color))

                        self.results_table.setItem(row_idx, column_indices[key], item)
            logger.debug(f"FileScanWindow: 成功填充 {display_count} 行搜索结果到表格。")

        except Exception as e:
            logger.critical(f"FileScanWindow: 更新搜索结果表格时发生错误: {e}", exc_info=True)
            QMessageBox.critical(self, "显示结果错误", f"无法显示搜索结果。\n错误: {e}")
        finally:
            logger.debug("FileScanWindow: 重新启用表格排序。")

    def _get_selected_path(self):
        selected_items = self.results_table.selectedItems()
        if selected_items:
            path = selected_items[0].data(Qt.ItemDataRole.UserRole)
            logger.debug(f"FileScanWindow: 获取选中路径: {path}")
            return path
        logger.debug("FileScanWindow: 未选中任何路径。")
        return None

    def _open_selected_item(self):
        full_path = self._get_selected_path()
        if full_path:
            logger.info(f"FileScanWindow: 尝试打开选中项: {full_path}")
            if os.path.exists(full_path):
                QDesktopServices.openUrl(QUrl.fromLocalFile(full_path))
                logger.info(f"FileScanWindow: 成功打开文件/文件夹: {full_path}")
            else:
                QMessageBox.warning(self, "文件不存在", f"路径不存在，建议重新索引:\n{full_path}")
                logger.warning(f"FileScanWindow: 尝试打开的文件/文件夹不存在: {full_path}")

    def _delete_selected_items(self):
        selected_paths = set()
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选择", "请先选择要删除的文件或文件夹。")
            return

        for item in selected_items:
            path = item.data(Qt.ItemDataRole.UserRole)
            if path:
                selected_paths.add(path)

        paths_to_delete = list(selected_paths)
        if not paths_to_delete:
            return

        item_count = len(paths_to_delete)
        preview_paths = "\n".join(paths_to_delete[:5])
        if item_count > 5:
            preview_paths += "\n..."

        reply = QMessageBox.question(
            self,
            "确认删除",
            f"您确定要永久删除选中的 {item_count} 个项目吗？\n\n{preview_paths}\n\n此操作无法撤销！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            db_paths_to_delete = []
            errors = []

            self.status_bar.showMessage(f"正在删除 {item_count} 个项目...")
            QApplication.processEvents()

            for path in paths_to_delete:
                try:
                    if not os.path.exists(path):
                        logger.warning(f"路径 {path} 在删除前已不存在，跳过。")
                        continue

                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f"已成功删除文件夹: {path}")
                    else:
                        os.remove(path)
                        logger.info(f"已成功删除文件: {path}")

                    db_paths_to_delete.append(path.lower())
                except Exception as e:
                    logger.error(f"删除 {path} 失败: {e}", exc_info=True)
                    errors.append(f"{os.path.basename(path)}: {e}")

            if db_paths_to_delete:
                self.db_manager.delete_files_batch(db_paths_to_delete)

            if errors:
                QMessageBox.critical(self, "删除时发生错误", "部分项目删除失败:\n\n" + "\n".join(errors))

            self.status_bar.showMessage(f"操作完成。成功删除 {len(db_paths_to_delete)} 个项目。", 5000)
            self._start_search()

    def _show_context_menu(self, pos):
        full_path = self._get_selected_path()
        if not full_path:
            logger.debug("FileScanWindow: 右键菜单：无选中项。")
            return

        menu = QMenu(self)
        open_action = menu.addAction("打开")
        open_folder_action = menu.addAction("打开所在文件夹")
        copy_path_action = menu.addAction("复制完整路径")

        menu.addSeparator()
        delete_action = menu.addAction("删除")

        action = menu.exec(self.results_table.mapToGlobal(pos))
        logger.debug(f"FileScanWindow: 右键菜单动作: {action and action.text()}")

        if action == open_action:
            self._open_selected_item()
        elif action == open_folder_action:
            folder = os.path.dirname(full_path)
            logger.info(f"FileScanWindow: 尝试打开所在文件夹: {folder}")
            if os.path.exists(folder):
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
                logger.info(f"FileScanWindow: 成功打开文件夹: {folder}")
            else:
                QMessageBox.warning(self, "文件夹不存在", f"路径不存在:\n{folder}")
                logger.warning(f"FileScanWindow: 尝试打开的文件夹不存在: {folder}")
        elif action == copy_path_action:
            QApplication.clipboard().setText(full_path)
            self.status_bar.showMessage("完整路径已复制到剪贴板。", 2000)
            logger.info(f"FileScanWindow: 完整路径 '{full_path}' 已复制到剪贴板。")
        elif action == delete_action:
            self._delete_selected_items()

    def closeEvent(self, event):
        logger.info("FileScanWindow: 应用程序正在关闭...")
        self.auto_index_timer.stop()
        logger.debug("FileScanWindow: 自动索引计时器已停止。")

        if self.indexer_thread and self.indexer_thread.isRunning():
            logger.info("FileScanWindow: 停止索引线程。")
            self.indexer_thread.stop()
            self.indexer_thread.wait(5000)
            if self.indexer_thread.isRunning():
                logger.warning("FileScanWindow: 索引线程未能按时停止。")

        if self.current_search_worker and self.current_search_worker.isRunning():
            logger.info("FileScanWindow: 停止搜索线程。")
            self.current_search_worker.stop()
            self.current_search_worker.wait(1000)
            if self.current_search_worker.isRunning():
                logger.warning("FileScanWindow: 搜索线程未能按时停止。")

        if self.cleanup_thread and self.cleanup_thread.isRunning():
            logger.info("FileScanWindow: 停止清理线程。")
            self.cleanup_thread.stop()
            self.cleanup_thread.wait(5000)
            if self.cleanup_thread.isRunning():
                logger.warning("FileScanWindow: 清理线程未能按时停止。")

        self.db_manager.close()
        logger.info("FileScanWindow: 后台线程已停止，数据库已关闭。程序退出。")
        super().closeEvent(event)


# --- 主程序入口 ---
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("FileScanApp")
        logger.info("应用程序 QApplication 实例创建成功。")
        window = FileScanWindow()
        window.show()
        logger.info("主窗口显示。")
        sys.exit(app.exec())
    except Exception as e:
        logger.critical(f"程序主循环发生致命错误: {e}", exc_info=True)
        QMessageBox.critical(None, "致命错误", f"程序遇到无法恢复的错误，即将退出。\n详情请查看日志文件。\n\n{e}")
        sys.exit(1)
