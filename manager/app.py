#!/usr/bin/env python3
"""
Process Manager Application
Manages multiple services (Python apps, PowerShell scripts) across different instances.
"""

import json
import os
import subprocess
import psutil
import threading
import logging
import time
import traceback
import atexit
import shlex
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

from flask import Flask, render_template, request, jsonify, send_from_directory

# ============================================================================
# Configuration and Setup
# ============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MANAGER_ROOT = Path(__file__).parent
PROJECT_ROOT = MANAGER_ROOT.parent
INSTANCES_DIR = MANAGER_ROOT / 'instances'
INSTANCE_SERVICE_CONFIGS_DIR = MANAGER_ROOT / 'instance_service_configs'
CONFIG_FILE = MANAGER_ROOT / 'config.json'

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ServiceProcess:
    """Represents a running service process."""
    service_id: str
    service_name: str
    pid: Optional[int] = None
    status: str = "stopped"  # stopped, running, error
    start_time: Optional[str] = None
    log_buffer: deque = None
    process: Optional[subprocess.Popen] = None
    
    def __post_init__(self):
        if self.log_buffer is None:
            self.log_buffer = deque(maxlen=500)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        uptime_seconds = self._calculate_uptime_seconds() if self.start_time else 0
        return {
            'service_id': self.service_id,
            'service_name': self.service_name,
            'pid': self.pid,
            'status': self.status,
            'start_time': self.start_time,
            'uptime': uptime_seconds,
            'uptime_seconds': uptime_seconds,
            'uptime_text': self._format_uptime(uptime_seconds) if uptime_seconds else None
        }
    
    def _calculate_uptime_seconds(self) -> int:
        """Calculate uptime in seconds from start_time."""
        if not self.start_time:
            return 0
        try:
            start = datetime.fromisoformat(self.start_time)
            uptime = datetime.now() - start
            return max(0, int(uptime.total_seconds()))
        except:
            return 0

    def _format_uptime(self, seconds: int) -> str:
        """Format uptime seconds to human readable string."""
        try:
            hours, remainder = divmod(seconds, 3600)
            minutes, secs = divmod(remainder, 60)
            return f"{hours}h {minutes}m {secs}s"
        except:
            return None

# ============================================================================
# Process Manager Class
# ============================================================================

class ProcessManager:
    """Manages lifecycle of multiple service processes."""
    
    def __init__(self):
        self.processes: Dict[str, ServiceProcess] = {}
        self.config: Dict = self._load_config()
        self.current_instance: str = "default"
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.should_exit = False
        self._network_snapshot: Optional[Dict[str, Any]] = None
        self._gpu_cache: Dict[str, Any] = {'timestamp': 0.0, 'data': {'available': False, 'utilization_percent': 0.0, 'memory_used_mb': 0.0, 'memory_total_mb': 0.0}}
        self._gpu_process_cache: Dict[str, Any] = {'timestamp': 0.0, 'data': {}}
        self._process_cpu_snapshots: Dict[int, Dict[str, float]] = {}

        # Prime psutil CPU measurement so next readings are meaningful
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        
        self._start_monitor()
    
    def _load_config(self) -> Dict:
        """Load configuration from config.json."""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return {}
    
    def _save_config(self):
        """Save configuration to config.json."""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _resolve_service_raw_path(self, service_path: str) -> Path:
        """Resolve service path relative to project/manager roots."""
        if service_path.startswith('..'):
            return (PROJECT_ROOT / service_path[3:]).resolve()
        return (MANAGER_ROOT / service_path).resolve()

    def _resolve_service_port(self, service_config: Dict, raw_path: Path) -> Optional[int]:
        """Resolve runtime port from service config or local config.json."""
        port_value = service_config.get('port')
        if isinstance(port_value, int):
            return port_value
        if isinstance(port_value, str) and port_value.isdigit():
            return int(port_value)

        # Fallback: service-local config.json -> server.port
        try:
            service_dir = raw_path if raw_path.is_dir() else raw_path.parent
            service_config_file = service_dir / 'config.json'
            if service_config_file.exists():
                with open(service_config_file, 'r', encoding='utf-8') as f:
                    service_local_config = json.load(f)
                local_port = service_local_config.get('server', {}).get('port')
                if isinstance(local_port, int):
                    return local_port
                if isinstance(local_port, str) and local_port.isdigit():
                    return int(local_port)
        except Exception as e:
            logger.warning(f"Could not resolve service port: {e}")

        return None

    def _kill_process_on_port(self, port: int) -> Dict[str, Any]:
        """Kill listening processes on a given TCP port."""
        killed_pids: List[int] = []
        errors: List[str] = []

        try:
            for conn in psutil.net_connections(kind='inet'):
                try:
                    if not conn.laddr or conn.laddr.port != port:
                        continue
                    if conn.status != psutil.CONN_LISTEN:
                        continue

                    pid = conn.pid
                    if not pid or pid in killed_pids:
                        continue

                    if pid == os.getpid():
                        continue

                    try:
                        proc = psutil.Process(pid)
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            proc.kill()
                            proc.wait(timeout=3)
                        killed_pids.append(pid)
                        logger.warning(f"Force-start: killed PID {pid} on port {port}")
                    except Exception as kill_error:
                        errors.append(f"PID {pid}: {kill_error}")
                except Exception:
                    continue
        except Exception as e:
            errors.append(str(e))

        return {'killed_pids': killed_pids, 'errors': errors}

    def _find_listening_pid_on_port(self, port: int) -> Optional[int]:
        """Find a listening process PID on a given port."""
        try:
            for conn in psutil.net_connections(kind='inet'):
                try:
                    if not conn.laddr or conn.laddr.port != port:
                        continue
                    if conn.status != psutil.CONN_LISTEN:
                        continue
                    if conn.pid and conn.pid != os.getpid():
                        return int(conn.pid)
                except Exception:
                    continue
        except Exception:
            return None
        return None

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Best-effort GPU stats (NVIDIA via nvidia-smi)."""
        now = time.time()
        cache_ttl = 3.0
        if (now - self._gpu_cache.get('timestamp', 0.0)) < cache_ttl:
            return self._gpu_cache.get('data', {})

        gpu_data = {
            'available': False,
            'utilization_percent': 0.0,
            'memory_used_mb': 0.0,
            'memory_total_mb': 0.0
        }

        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=1.2,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )

            if result.returncode == 0 and result.stdout:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if lines:
                    gpu_util_total = 0.0
                    mem_used_total = 0.0
                    mem_total_total = 0.0
                    for line in lines:
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) >= 3:
                            gpu_util_total += float(parts[0])
                            mem_used_total += float(parts[1])
                            mem_total_total += float(parts[2])

                    gpu_data = {
                        'available': True,
                        'utilization_percent': round(gpu_util_total / max(1, len(lines)), 2),
                        'memory_used_mb': round(mem_used_total, 2),
                        'memory_total_mb': round(mem_total_total, 2)
                    }
        except Exception:
            pass

        self._gpu_cache = {'timestamp': now, 'data': gpu_data}
        return gpu_data

    def _get_gpu_process_stats(self) -> Dict[int, float]:
        """Best-effort GPU memory usage per PID (NVIDIA compute apps)."""
        now = time.time()
        cache_ttl = 3.0
        if (now - self._gpu_process_cache.get('timestamp', 0.0)) < cache_ttl:
            return self._gpu_process_cache.get('data', {})

        gpu_pid_map: Dict[int, float] = {}
        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=1.2,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            )

            if result.returncode == 0 and result.stdout:
                lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) < 2:
                        continue
                    try:
                        pid = int(parts[0])
                        gpu_mem_mb = float(parts[1])
                        gpu_pid_map[pid] = round(gpu_mem_mb, 2)
                    except Exception:
                        continue
        except Exception:
            pass

        self._gpu_process_cache = {'timestamp': now, 'data': gpu_pid_map}
        return gpu_pid_map

    def _get_network_meta(self) -> Dict[str, Any]:
        """Return system network counters and transfer rates."""
        counters = psutil.net_io_counters()
        now = time.time()

        upload_bps = 0.0
        download_bps = 0.0
        if self._network_snapshot:
            elapsed = max(0.001, now - self._network_snapshot.get('timestamp', now))
            upload_bps = max(0.0, (counters.bytes_sent - self._network_snapshot.get('bytes_sent', counters.bytes_sent)) / elapsed)
            download_bps = max(0.0, (counters.bytes_recv - self._network_snapshot.get('bytes_recv', counters.bytes_recv)) / elapsed)

        self._network_snapshot = {
            'timestamp': now,
            'bytes_sent': counters.bytes_sent,
            'bytes_recv': counters.bytes_recv
        }

        return {
            'bytes_sent': counters.bytes_sent,
            'bytes_recv': counters.bytes_recv,
            'upload_bps': round(upload_bps, 2),
            'download_bps': round(download_bps, 2)
        }

    def _collect_service_metrics(self, service_process: ServiceProcess) -> Dict[str, Any]:
        """Collect runtime metrics for a single service process."""
        gpu_stats = self._get_gpu_stats()
        gpu_pid_map = self._get_gpu_process_stats()
        cpu_count = max(int(psutil.cpu_count(logical=True) or 1), 1)
        sample_time = time.time()

        metrics = {
            'cpu': 0.0,
            'cpu_raw': 0.0,
            'memory': 0.0,
            'memory_percent': 0.0,
            'gpu': 0.0,
            'gpu_memory_mb': 0.0,
            'gpu_available': bool(gpu_stats.get('available', False)),
            'threads': 0,
            'handles': 0,
            'connections': 0,
            'io_read_mb': 0.0,
            'io_write_mb': 0.0,
            'uptime': 0,
            'uptime_seconds': 0,
            'uptime_text': '--'
        }

        if not service_process or not service_process.pid or service_process.status != 'running':
            return metrics

        try:
            root_proc = psutil.Process(service_process.pid)
            proc_tree = [root_proc]
            try:
                proc_tree.extend(root_proc.children(recursive=True))
            except Exception:
                pass

            alive_procs: List[psutil.Process] = []
            proc_pids = set()
            for current_proc in proc_tree:
                try:
                    if not current_proc.is_running():
                        continue
                    pid = int(current_proc.pid)
                    if pid in proc_pids:
                        continue
                    proc_pids.add(pid)
                    alive_procs.append(current_proc)
                except Exception:
                    continue

            if not alive_procs:
                return metrics

            cpu_raw_total = 0.0
            memory_mb_total = 0.0
            memory_percent_total = 0.0
            threads_total = 0
            handles_total = 0
            connections_total = 0
            io_read_total = 0.0
            io_write_total = 0.0

            for current_proc in alive_procs:
                try:
                    with current_proc.oneshot():
                        mem_info = current_proc.memory_info()
                        memory_mb_total += float(mem_info.rss) / (1024 * 1024)
                        memory_percent_total += float(current_proc.memory_percent() or 0.0)
                        threads_total += int(current_proc.num_threads())
                        if hasattr(current_proc, 'num_handles'):
                            handles_total += int(current_proc.num_handles())

                        cpu_times = current_proc.cpu_times()
                        proc_cpu_time = float((cpu_times.user or 0.0) + (cpu_times.system or 0.0))
                        previous = self._process_cpu_snapshots.get(current_proc.pid)
                        if previous:
                            elapsed = max(0.001, sample_time - float(previous.get('timestamp', sample_time)))
                            cpu_delta = max(0.0, proc_cpu_time - float(previous.get('cpu_time', proc_cpu_time)))
                            cpu_raw_total += (cpu_delta / elapsed) * 100.0
                        self._process_cpu_snapshots[current_proc.pid] = {
                            'timestamp': sample_time,
                            'cpu_time': proc_cpu_time
                        }

                    try:
                        net_conns = current_proc.net_connections(kind='inet')
                        connections_total += len(net_conns)
                    except Exception:
                        pass

                    try:
                        io = current_proc.io_counters()
                        io_read_total += float(io.read_bytes)
                        io_write_total += float(io.write_bytes)
                    except Exception:
                        pass
                except Exception:
                    continue

            # Cleanup stale PID snapshots
            active_pid_set = set(proc_pids)
            for tracked_pid in list(self._process_cpu_snapshots.keys()):
                if tracked_pid not in active_pid_set:
                    try:
                        if not psutil.pid_exists(tracked_pid):
                            self._process_cpu_snapshots.pop(tracked_pid, None)
                    except Exception:
                        self._process_cpu_snapshots.pop(tracked_pid, None)

            service_gpu_mem = sum(float(gpu_pid_map.get(pid, 0.0) or 0.0) for pid in proc_pids)
            total_gpu_mem_used = float(gpu_stats.get('memory_used_mb', 0.0) or 0.0)
            global_gpu_util = float(gpu_stats.get('utilization_percent', 0.0) or 0.0)

            metrics['cpu_raw'] = round(cpu_raw_total, 2)
            metrics['cpu'] = round(min(100.0, cpu_raw_total / cpu_count), 2)
            metrics['memory'] = round(memory_mb_total, 2)
            metrics['memory_percent'] = round(memory_percent_total, 2)
            metrics['threads'] = threads_total
            metrics['handles'] = handles_total
            metrics['connections'] = connections_total
            metrics['io_read_mb'] = round(io_read_total / (1024 * 1024), 2)
            metrics['io_write_mb'] = round(io_write_total / (1024 * 1024), 2)
            metrics['gpu_memory_mb'] = round(service_gpu_mem, 2)

            if service_gpu_mem > 0 and metrics['gpu_available']:
                if total_gpu_mem_used > 0:
                    gpu_share = min(1.0, service_gpu_mem / total_gpu_mem_used)
                    metrics['gpu'] = round(global_gpu_util * gpu_share, 2)
                else:
                    metrics['gpu'] = round(global_gpu_util, 2)

            uptime_seconds = service_process._calculate_uptime_seconds()
            metrics['uptime'] = uptime_seconds
            metrics['uptime_seconds'] = uptime_seconds
            metrics['uptime_text'] = service_process._format_uptime(uptime_seconds) if uptime_seconds else '--'
        except Exception:
            pass

        return metrics

    def get_instance_meta(self, statuses: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate instance meta metrics from service statuses and host stats."""
        running_services = [s for s in statuses.values() if s.get('status') == 'running']

        total_service_cpu = round(sum(float(s.get('cpu', 0.0) or 0.0) for s in running_services), 2)
        total_service_cpu_raw = round(sum(float(s.get('cpu_raw', s.get('cpu', 0.0)) or 0.0) for s in running_services), 2)
        max_service_cpu = round(max((float(s.get('cpu', 0.0) or 0.0) for s in running_services), default=0.0), 2)
        total_service_memory_mb = round(sum(float(s.get('memory', 0.0) or 0.0) for s in running_services), 2)
        max_uptime = max((int(s.get('uptime_seconds', 0) or 0) for s in running_services), default=0)

        try:
            host_cpu = round(float(psutil.cpu_percent(interval=0.05) or 0.0), 2)
        except Exception:
            host_cpu = 0.0

        network = self._get_network_meta()
        gpu = self._get_gpu_stats()

        return {
            'running_services': len(running_services),
            'total_services': len(statuses),
            'service_cpu_percent': total_service_cpu,
            'service_cpu_raw_percent': total_service_cpu_raw,
            'service_cpu_percent_max': max_service_cpu,
            'service_memory_mb': total_service_memory_mb,
            'uptime_seconds': max_uptime,
            'host_cpu_percent': host_cpu,
            'gpu': gpu,
            'network': network
        }
    
    def start_service(self, service_id: str, instance: str = None, force: bool = False) -> Dict:
        """Start a service process."""
        instance = instance or self.current_instance
        
        try:
            with self.lock:
                if service_id in self.processes and self.is_running(service_id):
                    if not force:
                        return {'success': False, 'error': 'Service already running'}
                    # Force mode: stop own tracked process first
                    service_process = self.processes.get(service_id)
                    if service_process and service_process.process:
                        try:
                            service_process.process.terminate()
                            try:
                                service_process.process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                service_process.process.kill()
                                service_process.process.wait(timeout=3)
                        except Exception as e:
                            logger.warning(f"Force-start failed to stop existing process: {e}")
                        service_process.status = 'stopped'
                        service_process.pid = None
                        service_process.process = None
                
                service_config = self.config.get('services', {}).get(service_id)
                if not service_config:
                    return {'success': False, 'error': 'Service configuration not found'}
                
                # Prepare working directory and command
                service_path = service_config.get('path', '')
                service_type = service_config.get('type', 'python')
                script_name = service_config.get('script', 'app.py')

                # Resolve raw path relative to project root
                raw_path = self._resolve_service_raw_path(service_path)

                if not raw_path.exists():
                    return {'success': False, 'error': f'Service path not found: {raw_path}'}

                # Apply per-instance config.json before service startup
                if instance and not apply_instance_service_config_to_runtime(instance, service_id):
                    return {
                        'success': False,
                        'error': f"Failed to apply instance config for '{service_id}' in instance '{instance}'"
                    }

                force_port_result = None
                if force:
                    target_port = self._resolve_service_port(service_config, raw_path)
                    if target_port:
                        force_port_result = self._kill_process_on_port(target_port)
                        if force_port_result.get('errors') and not force_port_result.get('killed_pids'):
                            return {
                                'success': False,
                                'error': f"Port {target_port} could not be cleared: {'; '.join(force_port_result['errors'])}"
                            }
                
                # Build command
                if service_type == 'python':
                    # Python services expect a directory path + script name
                    work_dir = raw_path if raw_path.is_dir() else raw_path.parent
                    script_path = work_dir / script_name
                    if not script_path.exists():
                        return {'success': False, 'error': f'{script_name} not found in {work_dir}'}

                    python_executable = service_config.get('python_executable', '').strip()
                    if python_executable:
                        py_path = Path(python_executable)
                        if not py_path.is_absolute():
                            py_path = (PROJECT_ROOT / py_path).resolve()
                        else:
                            py_path = py_path.resolve()

                        if not py_path.exists():
                            return {'success': False, 'error': f'Python executable not found: {py_path}'}

                        cmd = [str(py_path), script_name]
                    else:
                        cmd = ['python', script_name]
                elif service_type == 'powershell':
                    # PowerShell can be configured either as direct script file path or directory + script
                    if raw_path.is_file() and raw_path.suffix.lower() == '.ps1':
                        script_path = raw_path
                        work_dir = raw_path.parent
                    else:
                        work_dir = raw_path if raw_path.is_dir() else raw_path.parent
                        if script_name:
                            script_path = work_dir / script_name
                        else:
                            script_path = work_dir / f"{work_dir.name}.ps1"

                    if not script_path.exists():
                        return {'success': False, 'error': f'PowerShell script not found: {script_path}'}
                    cmd = [
                        'powershell',
                        '-NoProfile',
                        '-ExecutionPolicy',
                        'Bypass',
                        '-File',
                        str(script_path)
                    ]
                else:
                    return {'success': False, 'error': f'Unknown service type: {service_type}'}
                
                # Create process
                try:
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(work_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )

                    # Detect immediate startup failure to avoid false "success"
                    time.sleep(0.35)
                    exit_code = process.poll()
                    if exit_code is not None:
                        startup_logs = []
                        try:
                            if process.stdout:
                                startup_logs = process.stdout.read().strip().splitlines()[-6:]
                        except Exception:
                            startup_logs = []

                        error_suffix = f' (exit code: {exit_code})'
                        if startup_logs:
                            error_suffix += f" | Logs: {' | '.join(startup_logs)}"

                        return {
                            'success': False,
                            'error': f'Service exited immediately{error_suffix}'
                        }
                    
                    service_process = self.processes.get(service_id)
                    if not service_process:
                        service_name = service_config.get('name', service_id)
                        service_process = ServiceProcess(
                            service_id=service_id,
                            service_name=service_name
                        )
                        self.processes[service_id] = service_process
                    
                    service_process.process = process
                    service_process.pid = process.pid
                    service_process.status = 'running'
                    service_process.start_time = datetime.now().isoformat()
                    
                    # Start log reader thread
                    log_thread = threading.Thread(
                        target=self._read_logs,
                        args=(service_id, process),
                        daemon=True
                    )
                    log_thread.start()
                    
                    logger.info(f"Started service '{service_id}' (PID: {process.pid})")
                    response = {
                        'success': True,
                        'service_id': service_id,
                        'pid': process.pid,
                        'start_time': service_process.start_time
                    }

                    if force and force_port_result is not None:
                        response['force_start'] = {
                            'port_cleared': True,
                            'killed_pids': force_port_result.get('killed_pids', []),
                            'errors': force_port_result.get('errors', [])
                        }

                    return response
                    
                except Exception as e:
                    logger.error(f"Error creating process: {e}")
                    return {'success': False, 'error': f'Failed to start process: {str(e)}'}
                    
        except Exception as e:
            logger.error(f"Error starting service '{service_id}': {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def stop_service(self, service_id: str) -> Dict:
        """Stop a running service."""
        try:
            with self.lock:
                service_process = self.processes.get(service_id)
                if not service_process:
                    return {'success': False, 'error': 'Service not found'}
                
                if not self.is_running(service_id):
                    service_process.status = 'stopped'
                    service_process.pid = None
                    return {'success': True, 'message': 'Service already stopped'}
                
                try:
                    service_process.process.terminate()
                    try:
                        service_process.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        service_process.process.kill()
                        service_process.process.wait()
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")
                
                service_process.status = 'stopped'
                service_process.pid = None
                service_process.process = None
                
                logger.info(f"Stopped service '{service_id}'")
                return {'success': True, 'service_id': service_id}
                
        except Exception as e:
            logger.error(f"Error stopping service '{service_id}': {e}")
            return {'success': False, 'error': str(e)}
    
    def restart_service(self, service_id: str, instance: str = None) -> Dict:
        """Restart a service."""
        instance = instance or self.current_instance
        
        try:
            # Stop the service
            stop_result = self.stop_service(service_id)
            if not stop_result.get('success'):
                return stop_result
            
            time.sleep(1)  # Wait a moment before restarting
            
            # Start the service
            return self.start_service(service_id, instance)
            
        except Exception as e:
            logger.error(f"Error restarting service '{service_id}': {e}")
            return {'success': False, 'error': str(e)}
    
    def is_running(self, service_id: str) -> bool:
        """Check if a service is running."""
        service_process = self.processes.get(service_id)
        if not service_process or not service_process.process:
            return False
        
        poll = service_process.process.poll()
        return poll is None
    
    def get_status(self, service_id: str) -> Dict:
        """Get status of a service."""
        service_process = self.processes.get(service_id)
        if not service_process:
            # Initialize service process if not exists
            service_config = self.config.get('services', {}).get(service_id)
            if service_config:
                service_process = ServiceProcess(
                    service_id=service_id,
                    service_name=service_config.get('name', service_id),
                    status='stopped'
                )
                self.processes[service_id] = service_process
            else:
                return {'success': False, 'error': 'Service not found'}

        # Discover running process by port if manager was restarted and process handle is missing
        if not service_process.process and not service_process.pid:
            service_config = self.config.get('services', {}).get(service_id, {})
            service_path = service_config.get('path', '')
            if service_path:
                raw_path = self._resolve_service_raw_path(service_path)
                discovered_port = self._resolve_service_port(service_config, raw_path)
                if discovered_port:
                    discovered_pid = self._find_listening_pid_on_port(discovered_port)
                    if discovered_pid:
                        service_process.pid = discovered_pid
                        service_process.status = 'running'
                        if not service_process.start_time:
                            try:
                                proc = psutil.Process(discovered_pid)
                                started = datetime.fromtimestamp(proc.create_time())
                                service_process.start_time = started.isoformat()
                            except Exception:
                                service_process.start_time = datetime.now().isoformat()
        
        # Update status
        if service_process.process:
            if service_process.process.poll() is not None:
                service_process.status = 'stopped'
                service_process.pid = None
                service_process.process = None
        elif service_process.pid:
            try:
                if not psutil.pid_exists(service_process.pid):
                    service_process.status = 'stopped'
                    service_process.pid = None
            except Exception:
                service_process.status = 'stopped'
                service_process.pid = None
        
        service_payload = service_process.to_dict()
        service_payload.update(self._collect_service_metrics(service_process))

        return {
            'success': True,
            'service': service_payload
        }
    
    def get_logs(self, service_id: str, lines: int = 100) -> Dict:
        """Get recent logs for a service."""
        service_process = self.processes.get(service_id)
        if not service_process:
            return {'success': False, 'error': 'Service not found'}
        
        logs = list(service_process.log_buffer)[-lines:] if service_process.log_buffer else []
        
        return {
            'success': True,
            'service_id': service_id,
            'logs': '\n'.join(logs)
        }
    
    def _read_logs(self, service_id: str, process: subprocess.Popen):
        """Read logs from a service process."""
        service_process = self.processes.get(service_id)
        if not service_process:
            return
        
        try:
            for line in process.stdout:
                if not line:
                    break
                line = line.strip()
                if line:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    formatted_line = f"[{timestamp}] {line}"
                    service_process.log_buffer.append(formatted_line)
                    logger.info(f"[{service_id}] {line}")
        except Exception as e:
            logger.error(f"Error reading logs for '{service_id}': {e}")
    
    def stop_all(self):
        """Stop all running services."""
        with self.lock:
            for service_id in list(self.processes.keys()):
                if self.is_running(service_id):
                    logger.info(f"Stopping service '{service_id}'...")
                    self.stop_service(service_id)
    
    def get_all_statuses(self) -> Dict:
        """Get status of all services."""
        statuses = {}
        for service_id in self.config.get('services', {}).keys():
            status_result = self.get_status(service_id)
            if status_result.get('success'):
                statuses[service_id] = status_result['service']
        return statuses
    
    def _start_monitor(self):
        """Start monitoring thread for zombie processes."""
        if self.monitor_thread is None:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """Monitor running processes and cleanup zombies."""
        while not self.should_exit:
            try:
                with self.lock:
                    for service_id, service_process in self.processes.items():
                        if service_process.process:
                            if service_process.process.poll() is not None:
                                service_process.status = 'stopped'
                                service_process.pid = None
                                service_process.process = None
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            if self.should_exit:
                break
    
    def shutdown(self):
        """Shutdown the process manager."""
        self.should_exit = True
        self.stop_all()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

# ============================================================================
# Global Process Manager Instance
# ============================================================================

pm = ProcessManager()

def cleanup():
    """Cleanup on application exit."""
    pm.shutdown()

atexit.register(cleanup)

# ============================================================================
# Instance Management Functions
# ============================================================================

def load_instances() -> List[Dict]:
    """Load all available instances."""
    instances = []
    if INSTANCES_DIR.exists():
        for instance_file in INSTANCES_DIR.glob('*.json'):
            try:
                with open(instance_file, 'r', encoding='utf-8') as f:
                    instance = json.load(f)
                    instance['filename'] = instance_file.stem
                    instances.append(instance)
            except Exception as e:
                logger.error(f"Error loading instance '{instance_file}': {e}")
    return instances

def get_instance(instance_name: str) -> Optional[Dict]:
    """Get a specific instance."""
    instance_file = INSTANCES_DIR / f"{instance_name}.json"
    if instance_file.exists():
        try:
            with open(instance_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading instance '{instance_name}': {e}")
    return None

def save_instance(instance_name: str, instance_data: Dict) -> bool:
    """Save an instance."""
    try:
        INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
        instance_file = INSTANCES_DIR / f"{instance_name}.json"
        with open(instance_file, 'w', encoding='utf-8') as f:
            json.dump(instance_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved instance '{instance_name}'")
        return True
    except Exception as e:
        logger.error(f"Error saving instance '{instance_name}': {e}")
        return False

def _to_int_port(value: Any) -> Optional[int]:
    """Convert a value to integer port if possible."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None

def extract_port_from_runtime_config(config: Dict) -> Optional[int]:
    """Extract service port from common config shapes."""
    if not isinstance(config, dict):
        return None

    direct = _to_int_port(config.get('port'))
    if direct:
        return direct

    server_port = _to_int_port((config.get('server') or {}).get('port'))
    if server_port:
        return server_port

    return None

def load_preview_config_for_instance_service(instance_name: str, service_id: str) -> Dict:
    """Load config for preview without forcing bootstrap side-effects."""
    cfg_file = get_instance_service_config_file(instance_name, service_id)
    if cfg_file.exists():
        try:
            with open(cfg_file, 'r', encoding='utf-8-sig') as f:
                parsed = json.load(f)
                if isinstance(parsed, dict):
                    return parsed
        except Exception as e:
            logger.warning(f"Failed to read instance config '{instance_name}/{service_id}' for preview: {e}")

    fallback = load_service_runtime_config(service_id)
    return fallback if isinstance(fallback, dict) else {}

def build_used_port_registry(exclude_instance: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """Collect all ports used by existing instances and map owners."""
    used_ports: Dict[str, List[Dict[str, str]]] = {}
    for instance in load_instances():
        instance_name = instance.get('filename')
        if not instance_name:
            continue
        if exclude_instance and instance_name == exclude_instance:
            continue

        services = instance.get('services') or {}
        for service_id, service_settings in services.items():
            if not isinstance(service_settings, dict) or not service_settings.get('enabled', False):
                continue

            config_data = load_preview_config_for_instance_service(instance_name, service_id)
            port = extract_port_from_runtime_config(config_data)
            if not port:
                continue

            port_key = str(port)
            used_ports.setdefault(port_key, []).append({
                'instance': instance_name,
                'service': service_id,
            })

    return used_ports


# Audio device config paths per service
AUDIO_CONFIG_PATHS = {
    'main_server': [{'type': 'input', 'path': ['microphone', 'device_name']}],
    'text_to_speech': [
        {'type': 'output', 'path': ['voicemod', 'output_name_substring']},
        {'type': 'output', 'path': ['voicemod', 'additional_outputs'], 'is_list': True},
    ],
}


def build_used_audio_registry(exclude_instance: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
    """Collect all audio devices used by existing instances.
    Returns dict keyed by device name (lowered) -> list of {instance, service, type, device}."""
    used_audio: Dict[str, List[Dict[str, str]]] = {}
    for instance in load_instances():
        instance_name = instance.get('filename')
        if not instance_name:
            continue
        if exclude_instance and instance_name == exclude_instance:
            continue

        services = instance.get('services') or {}
        for service_id, service_settings in services.items():
            if not isinstance(service_settings, dict) or not service_settings.get('enabled', False):
                continue
            audio_defs = AUDIO_CONFIG_PATHS.get(service_id)
            if not audio_defs:
                continue

            config_data = load_preview_config_for_instance_service(instance_name, service_id)

            for audio_def in audio_defs:
                # Walk nested path
                val = config_data
                for key in audio_def['path']:
                    if isinstance(val, dict):
                        val = val.get(key)
                    else:
                        val = None
                        break

                if val is None:
                    continue

                # Handle list values (additional_outputs)
                device_values = []
                if audio_def.get('is_list') and isinstance(val, list):
                    device_values = [v.strip() for v in val if isinstance(v, str) and v.strip()]
                elif isinstance(val, str) and val.strip():
                    device_values = [val.strip()]

                for dev_val in device_values:
                    device_key = dev_val.lower()
                    used_audio.setdefault(device_key, []).append({
                        'instance': instance_name,
                        'service': service_id,
                        'type': audio_def['type'],
                        'device': dev_val,
                    })

    return used_audio

def delete_instance(instance_name: str) -> Dict:
    """Delete an instance: stop all its services, remove config files, remove instance JSON."""
    result = {'stopped_services': [], 'errors': []}
    try:
        # 1. Load instance to know which services belong to it
        instance = get_instance(instance_name)

        # 2. If this is the currently active instance, stop all running services
        if pm.current_instance == instance_name:
            for service_id in list(pm.processes.keys()):
                if pm.is_running(service_id):
                    logger.info(f"Stopping service '{service_id}' (deleting instance '{instance_name}')")
                    stop_res = pm.stop_service(service_id)
                    if stop_res.get('success'):
                        result['stopped_services'].append(service_id)
                    else:
                        result['errors'].append(f"Failed to stop {service_id}: {stop_res.get('error', '?')}")
            pm.current_instance = 'default'

        # 3. Delete instance JSON file
        instance_file = INSTANCES_DIR / f"{instance_name}.json"
        if instance_file.exists():
            instance_file.unlink()

        # 4. Delete all per-instance service config files
        instance_cfg_dir = INSTANCE_SERVICE_CONFIGS_DIR / instance_name
        if instance_cfg_dir.exists() and instance_cfg_dir.is_dir():
            shutil.rmtree(instance_cfg_dir, ignore_errors=True)

        # 4b. Release virtual audio cable assignment
        try:
            va = load_virtual_audio_assignments()
            before = len(va['assignments'])
            va['assignments'] = [a for a in va['assignments'] if a['instance'] != instance_name]
            if len(va['assignments']) < before:
                save_virtual_audio_assignments(va)
                logger.info(f"Released virtual audio cable for deleted instance '{instance_name}'")
        except Exception as e:
            logger.warning(f"Could not release virtual audio for '{instance_name}': {e}")

        # 5. Verify deletion
        if instance_file.exists() or (instance_cfg_dir.exists() and any(instance_cfg_dir.iterdir())):
            result['success'] = False
            result['error'] = 'Some files could not be deleted'
            return result

        logger.info(f"Deleted instance '{instance_name}' (stopped: {result['stopped_services']})")
        result['success'] = True
        return result
    except Exception as e:
        logger.error(f"Error deleting instance '{instance_name}': {e}")
        result['success'] = False
        result['error'] = str(e)
        return result

def get_instance_service_config_dir(instance_name: str) -> Path:
    """Directory for per-instance service config files."""
    return INSTANCE_SERVICE_CONFIGS_DIR / instance_name

def get_instance_service_config_file(instance_name: str, service_id: str) -> Path:
    """Path for per-instance service config JSON."""
    return get_instance_service_config_dir(instance_name) / f"{service_id}.json"

def bootstrap_instance_service_config(instance_name: str, service_id: str) -> Optional[Path]:
    """Create per-instance service config from base runtime config if missing."""
    try:
        target = get_instance_service_config_file(instance_name, service_id)
        if target.exists():
            return target

        base_config = load_service_runtime_config(service_id)
        if base_config is None:
            base_config = {}

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(base_config, f, indent=2, ensure_ascii=False)
        return target
    except Exception as e:
        logger.error(f"Error bootstrapping instance config for '{instance_name}/{service_id}': {e}")
        return None

def apply_instance_service_config_to_runtime(instance_name: str, service_id: str) -> bool:
    """Apply per-instance config to service runtime config.json before starting service."""
    try:
        source_file = get_instance_service_config_file(instance_name, service_id)
        if not source_file.exists():
            source_file = bootstrap_instance_service_config(instance_name, service_id)
            if not source_file:
                return False

        runtime_file = resolve_service_config_file(service_id)
        if not runtime_file:
            # No runtime config.json in service folder -> nothing to apply
            return True

        with open(source_file, 'r', encoding='utf-8-sig') as src:
            config_data = json.load(src)

        with open(runtime_file, 'w', encoding='utf-8') as dst:
            json.dump(config_data, dst, indent=2, ensure_ascii=False)

        return True
    except Exception as e:
        logger.error(f"Error applying instance runtime config for '{instance_name}/{service_id}': {e}")
        return False

# ============================================================================
# Config Management Functions
# ============================================================================

def load_service_config(service_id: str) -> Optional[Dict]:
    """Load configuration for a specific service."""
    return pm.config.get('services', {}).get(service_id)

def save_service_config(service_id: str, config: Dict) -> bool:
    """Save configuration for a service."""
    try:
        if 'services' not in pm.config:
            pm.config['services'] = {}
        pm.config['services'][service_id] = config
        pm._save_config()
        return True
    except Exception as e:
        logger.error(f"Error saving service config: {e}")
        return False

def resolve_service_config_file(service_id: str) -> Optional[Path]:
    """Resolve runtime config.json path for a service."""
    try:
        service_cfg = pm.config.get('services', {}).get(service_id)
        if not service_cfg:
            return None

        service_path = service_cfg.get('path', '')
        if not service_path:
            return None

        raw_path = pm._resolve_service_raw_path(service_path)
        service_dir = raw_path if raw_path.is_dir() else raw_path.parent
        config_file = service_dir / 'config.json'
        return config_file if config_file.exists() else None
    except Exception as e:
        logger.error(f"Error resolving config file for service '{service_id}': {e}")
        return None

def load_service_runtime_config(service_id: str) -> Optional[Dict]:
    """Load service runtime config.json, fallback to manager config entry."""
    config_file = resolve_service_config_file(service_id)
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading runtime config for '{service_id}': {e}")
            return None

    # Fallback to manager-level service config
    return load_service_config(service_id)

def load_instance_service_runtime_config(instance_name: str, service_id: str) -> Optional[Dict]:
    """Load per-instance service config JSON; auto-bootstrap from base config when missing."""
    instance_file = get_instance_service_config_file(instance_name, service_id)
    if not instance_file.exists():
        instance_file = bootstrap_instance_service_config(instance_name, service_id)
        if not instance_file:
            return None

    try:
        with open(instance_file, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading instance runtime config for '{instance_name}/{service_id}': {e}")
        try:
            base_config = load_service_runtime_config(service_id)
            if base_config is None:
                base_config = {}
            instance_file.parent.mkdir(parents=True, exist_ok=True)
            with open(instance_file, 'w', encoding='utf-8') as f:
                json.dump(base_config, f, indent=2, ensure_ascii=False)
            return base_config
        except Exception as heal_error:
            logger.error(f"Error healing instance runtime config for '{instance_name}/{service_id}': {heal_error}")
            return None

def save_service_runtime_config(service_id: str, config: Dict) -> bool:
    """Save service runtime config.json, fallback to manager config entry."""
    config_file = resolve_service_config_file(service_id)
    if config_file:
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving runtime config for '{service_id}': {e}")
            return False

    return save_service_config(service_id, config)

def save_instance_service_runtime_config(instance_name: str, service_id: str, config: Dict) -> bool:
    """Save per-instance service config JSON."""
    try:
        target_file = get_instance_service_config_file(instance_name, service_id)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving instance runtime config for '{instance_name}/{service_id}': {e}")
        return False

# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/')
def index():
    """Main page - Instance selector."""
    try:
        # Load all instances
        instances = load_instances()
        current_instance_name = pm.current_instance or 'default'
        
        return render_template(
            'index.html',
            instances=instances,
            current_instance_name=current_instance_name
        )
    except Exception as e:
        logger.error(f"Error in index: {e}")
        return f"Error: {str(e)}", 500

@app.route('/instance/<instance_name>/dashboard')
def instance_dashboard(instance_name):
    """Instance dashboard - Shows statistics, monitoring, and system info."""
    try:
        # Load instance
        instance = get_instance(instance_name)
        if not instance:
            # Try to load from instances list
            instances = load_instances()
            instance = next((i for i in instances if i.get('filename') == instance_name), None)
            if not instance:
                return f"Instance '{instance_name}' not found", 404
        else:
            instance['filename'] = instance_name
        
        # Set as current
        pm.current_instance = instance_name
        
        # Load all instances for navigation
        instances = load_instances()
        
        # Get all services config
        all_services = pm.config.get('services', {})
        
        # Filter services for this instance
        if instance.get('services'):
            instance_service_ids = set(instance['services'].keys())
            services = {k: v for k, v in all_services.items() if k in instance_service_ids}
            service_config = instance['services']
        else:
            services = all_services
            service_config = {}
        
        # Get status of all services
        statuses = pm.get_all_statuses()
        
        return render_template(
            'dashboard.html',
            instance=instance,
            instance_name=instance_name,
            instances=instances,
            services=services,
            service_config=service_config,
            statuses=statuses,
            config={
                'services': pm.config.get('services', {}),
                'manager': pm.config.get('manager', {})
            }
        )
    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        return f"Error: {str(e)}", 500

@app.route('/instance/<instance_name>/services')
def instance_services(instance_name):
    """Instance services management - Manage and control services."""
    try:
        # Load instance
        instance = get_instance(instance_name)
        if not instance:
            # Try to load from instances list
            instances = load_instances()
            instance = next((i for i in instances if i.get('filename') == instance_name), None)
            if not instance:
                return f"Instance '{instance_name}' not found", 404
        else:
            instance['filename'] = instance_name
        
        # Set as current
        pm.current_instance = instance_name
        
        # Load all instances for navigation
        instances = load_instances()
        
        # Get all services config
        all_services = pm.config.get('services', {})
        
        # Filter services for this instance
        if instance.get('services'):
            instance_service_ids = set(instance['services'].keys())
            services = {k: v for k, v in all_services.items() if k in instance_service_ids}
            service_config = instance['services']
        else:
            services = all_services
            service_config = {}
        
        # Get status of all services
        statuses = pm.get_all_statuses()
        
        return render_template(
            'services_management.html',
            instance=instance,
            instance_name=instance_name,
            instances=instances,
            services=services,
            service_config=service_config,
            statuses=statuses,
            config={
                'services': pm.config.get('services', {}),
                'manager': pm.config.get('manager', {})
            }
        )
    except Exception as e:
        logger.error(f"Error in services: {e}")
        return f"Error: {str(e)}", 500

@app.route('/instance/<instance_name>/config')
def instance_config(instance_name):
    """Instance config page - edit runtime config.json per service."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            instances = load_instances()
            instance = next((i for i in instances if i.get('filename') == instance_name), None)
            if not instance:
                return f"Instance '{instance_name}' not found", 404
        else:
            instance['filename'] = instance_name

        pm.current_instance = instance_name
        instances = load_instances()

        all_services = pm.config.get('services', {})
        if instance.get('services'):
            instance_service_ids = set(instance['services'].keys())
            services = {k: v for k, v in all_services.items() if k in instance_service_ids}
        else:
            services = all_services

        return render_template(
            'instance_config.html',
            instance=instance,
            instance_name=instance_name,
            instances=instances,
            services=services
        )
    except Exception as e:
        logger.error(f"Error in instance config: {e}")
        return f"Error: {str(e)}", 500

@app.route('/configs')
def configs_page():
    """Configuration management page."""
    try:
        return render_template(
            'configs.html',
            config={
                'services': pm.config.get('services', {}),
                'manager': pm.config.get('manager', {}),
                'project_root': pm.config.get('project_root', '..')
            }
        )
    except Exception as e:
        logger.error(f"Error in configs page: {e}")
        return f"Error: {str(e)}", 500

# ============================================================================
# API Endpoints - Audio Devices
# ============================================================================

@app.route('/api/audio/devices', methods=['GET'])
def api_audio_devices():
    """List all available audio input and output devices."""
    VIRTUAL_PATTERNS = ['vb-audio', 'cable', 'voicemod', 'xsplit', 'virtual']
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        inputs = []
        outputs = []
        for i, d in enumerate(devs):
            name_lower = d['name'].lower()
            is_virtual = any(p in name_lower for p in VIRTUAL_PATTERNS)
            if d['max_input_channels'] > 0:
                inputs.append({'index': i, 'name': d['name'], 'channels': d['max_input_channels'], 'virtual': is_virtual})
            if d['max_output_channels'] > 0:
                outputs.append({'index': i, 'name': d['name'], 'channels': d['max_output_channels'], 'virtual': is_virtual})
        return jsonify({'success': True, 'inputs': inputs, 'outputs': outputs}), 200
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}")
        return jsonify({'success': False, 'error': str(e), 'inputs': [], 'outputs': []}), 200

# ============================================================================
# Virtual Audio Cable Assignment System
# ============================================================================

VIRTUAL_AUDIO_FILE = Path(__file__).parent / 'virtual_audio.json'

# Known virtual cable pair patterns:
# Each pair has an input endpoint (what you READ audio from) and an output endpoint (what you WRITE audio to)
CABLE_PAIR_PATTERNS = [
    {
        'id': 'vb-cable',
        'label': 'VB-Audio Virtual Cable',
        'read_include': ['cable output', 'vb-audio virtual'],
        'read_exclude': ['point'],
        'write_include': ['cable input', 'vb-audio virtual'],
        'write_exclude': ['16ch'],
    },
    {
        'id': 'vb-point',
        'label': 'VB-Audio Point',
        'read_include': ['vb-audio point'],
        'read_exclude': ['cable'],
        'write_include': ['output', 'vb-audio point'],
        'write_exclude': [],
    },
]


def load_virtual_audio_assignments():
    """Load virtual audio assignments from JSON file."""
    try:
        if VIRTUAL_AUDIO_FILE.exists():
            return json.loads(VIRTUAL_AUDIO_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        logger.error(f"Error loading virtual audio assignments: {e}")
    return {'assignments': [], 'counter': 0}


def save_virtual_audio_assignments(data):
    """Save virtual audio assignments to JSON file."""
    VIRTUAL_AUDIO_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def detect_available_cable_pairs():
    """Detect virtual audio cable pairs from system devices."""
    import sounddevice as sd
    devs = sd.query_devices()

    # Collect ALL devices, then prefer longest name per match (avoids MME truncation)
    inputs = []
    outputs = []
    for i, d in enumerate(devs):
        name = d['name']
        if d['max_input_channels'] > 0:
            inputs.append({'index': i, 'name': name})
        if d['max_output_channels'] > 0:
            outputs.append({'index': i, 'name': name})

    pairs = []
    for pattern in CABLE_PAIR_PATTERNS:
        # Find matching input device (read from) — prefer longest name
        read_candidates = []
        for inp in inputs:
            nl = inp['name'].lower()
            if all(p in nl for p in pattern['read_include']):
                if pattern.get('read_exclude') and any(e in nl for e in pattern['read_exclude']):
                    continue
                read_candidates.append(inp)
        read_dev = max(read_candidates, key=lambda d: len(d['name']), default=None)

        # Find matching output device (write to) — prefer longest name
        write_candidates = []
        for outp in outputs:
            nl = outp['name'].lower()
            if all(p in nl for p in pattern['write_include']):
                if pattern.get('write_exclude') and any(e in nl for e in pattern['write_exclude']):
                    continue
                write_candidates.append(outp)
        write_dev = max(write_candidates, key=lambda d: len(d['name']), default=None)

        if read_dev and write_dev:
            pairs.append({
                'id': pattern['id'],
                'label': pattern['label'],
                'read_from': read_dev['name'],
                'write_to': write_dev['name'],
            })

    return pairs


@app.route('/api/audio/virtual/assign', methods=['POST'])
def api_virtual_audio_assign():
    """Assign a virtual audio cable pair to an instance (idempotent)."""
    data = request.get_json() or {}
    instance_name = data.get('instance_name', '').strip()
    if not instance_name:
        return jsonify({'success': False, 'error': 'instance_name required'}), 400

    va = load_virtual_audio_assignments()

    # Check if already assigned — return existing (idempotent)
    existing = next((a for a in va['assignments'] if a['instance'] == instance_name), None)
    if existing:
        return jsonify({'success': True, 'assignment': existing, 'already_existed': True})

    # Find available (unassigned) cable pairs
    all_pairs = detect_available_cable_pairs()
    assigned_cable_ids = {a.get('cable_id') for a in va['assignments']}

    available = None
    for pair in all_pairs:
        if pair['id'] not in assigned_cable_ids:
            available = pair
            break

    if not available:
        return jsonify({
            'success': False,
            'error': f'Keine freien virtuellen Audio-Kabel verfügbar '
                     f'({len(all_pairs)} erkannt, {len(assigned_cable_ids)} belegt). '
                     f'Bitte zusätzliche VB-Audio Cables installieren.'
        }), 400

    idx = va['counter']
    slot_name = f"ki_audio_experiment {idx} {instance_name}"

    assignment = {
        'slot_name': slot_name,
        'index': idx,
        'instance': instance_name,
        'cable_id': available['id'],
        'cable_label': available['label'],
        'read_from': available['read_from'],
        'write_to': available['write_to'],
    }

    va['assignments'].append(assignment)
    va['counter'] = idx + 1
    save_virtual_audio_assignments(va)

    logger.info(f"Virtual audio assigned: {slot_name} → {available['label']} "
                f"(read: {available['read_from']}, write: {available['write_to']})")

    return jsonify({'success': True, 'assignment': assignment, 'already_existed': False})


@app.route('/api/audio/virtual/release/<instance_name>', methods=['DELETE'])
def api_virtual_audio_release(instance_name):
    """Release a virtual audio cable pair from an instance."""
    va = load_virtual_audio_assignments()
    before = len(va['assignments'])
    va['assignments'] = [a for a in va['assignments'] if a['instance'] != instance_name]
    if len(va['assignments']) == before:
        return jsonify({'success': False, 'error': 'Kein Audio-Kabel für diese Instanz zugewiesen'}), 404
    save_virtual_audio_assignments(va)
    logger.info(f"Virtual audio released for instance '{instance_name}'")
    return jsonify({'success': True})


@app.route('/api/audio/virtual/list', methods=['GET'])
def api_virtual_audio_list():
    """List all virtual audio assignments and available cable pairs."""
    va = load_virtual_audio_assignments()
    try:
        pairs = detect_available_cable_pairs()
    except Exception:
        pairs = []
    assigned_cable_ids = {a.get('cable_id') for a in va['assignments']}
    return jsonify({
        'success': True,
        'assignments': va['assignments'],
        'available_pairs': pairs,
        'total_pairs': len(pairs),
        'assigned_count': len(va['assignments']),
        'free_count': len([p for p in pairs if p['id'] not in assigned_cable_ids]),
    })


# ============================================================================
# API Endpoints - Status and Logs (Global)
# ============================================================================

@app.route('/api/status/all', methods=['GET'])
def api_get_all_statuses():
    """Get all service statuses."""
    try:
        instance_name = request.args.get('instance', default=pm.current_instance, type=str)
        statuses = pm.get_all_statuses()

        # Optional instance scoping
        instance = get_instance(instance_name) if instance_name else None
        if instance and instance.get('services'):
            instance_service_ids = set(instance['services'].keys())
            statuses = {service_id: status for service_id, status in statuses.items() if service_id in instance_service_ids}

        instance_meta = pm.get_instance_meta(statuses)

        return jsonify({
            'success': True,
            'instance': instance_name,
            'services': statuses,
            'instance_meta': instance_meta
        }), 200
    except Exception as e:
        logger.error(f"API error (get_all_statuses): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API Endpoints - Instance Management
# ============================================================================

@app.route('/api/instance/list', methods=['GET'])
def api_list_instances():
    """List all instances."""
    try:
        instances = load_instances()
        return jsonify({'success': True, 'instances': instances}), 200
    except Exception as e:
        logger.error(f"API error (list_instances): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/save', methods=['POST'])
def api_save_instance():
    """Save an instance."""
    try:
        data = request.get_json() or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

        service_runtime_configs = data.pop('service_runtime_configs', {})
        if service_runtime_configs is None:
            service_runtime_configs = {}
        if not isinstance(service_runtime_configs, dict):
            return jsonify({'success': False, 'error': 'service_runtime_configs must be an object'}), 400

        instance_name = data.get('name')
        # Use filename if provided, otherwise sanitize the name
        filename = data.pop('filename', None) or instance_name
        
        if not instance_name:
            return jsonify({'success': False, 'error': 'Missing instance name'}), 400
        
        if save_instance(filename, data):
            failed_configs: List[str] = []
            services = data.get('services') or {}

            for service_id, cfg in service_runtime_configs.items():
                if service_id not in services:
                    continue
                if not isinstance(cfg, dict):
                    failed_configs.append(service_id)
                    continue
                if not save_instance_service_runtime_config(filename, service_id, cfg):
                    failed_configs.append(service_id)

            if failed_configs:
                return jsonify({
                    'success': False,
                    'error': 'Instance saved but some service configs failed',
                    'failed_services': failed_configs,
                }), 500

            return jsonify({'success': True, 'message': 'Instance saved'}), 200
        return jsonify({'success': False, 'error': 'Failed to save instance'}), 500
    except Exception as e:
        logger.error(f"API error (save_instance): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/configs/preview', methods=['POST'])
def api_instance_configs_preview():
    """Return editable service configs plus used-port ownership info for create/edit modal."""
    try:
        data = request.get_json() or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

        services_payload = data.get('services') or {}
        editing_instance = data.get('editing_instance')

        enabled_service_ids: List[str] = []
        if isinstance(services_payload, list):
            for item in services_payload:
                if isinstance(item, str) and item.strip():
                    enabled_service_ids.append(item.strip())
        elif isinstance(services_payload, dict):
            for service_id, settings in services_payload.items():
                if isinstance(settings, dict):
                    enabled_service_ids.append(service_id)
        else:
            return jsonify({'success': False, 'error': 'services must be list or object'}), 400

        configs: Dict[str, Dict] = {}
        for service_id in enabled_service_ids:
            if editing_instance:
                cfg = load_preview_config_for_instance_service(editing_instance, service_id)
            else:
                cfg = load_service_runtime_config(service_id) or {}
            configs[service_id] = cfg if isinstance(cfg, dict) else {}

        used_ports = build_used_port_registry(exclude_instance=editing_instance)
        used_audio = build_used_audio_registry(exclude_instance=editing_instance)

        return jsonify({
            'success': True,
            'configs': configs,
            'used_ports': used_ports,
            'used_audio': used_audio,
            'editing_instance': editing_instance,
        }), 200
    except Exception as e:
        logger.error(f"API error (instance_configs_preview): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/delete/<instance_name>', methods=['DELETE'])
def api_delete_instance(instance_name):
    """Delete an instance: stop services, remove configs, remove instance."""
    try:
        result = delete_instance(instance_name)
        if result.get('success'):
            msg = f"Instance gelöscht"
            if result.get('stopped_services'):
                msg += f" (gestoppt: {', '.join(result['stopped_services'])})"
            return jsonify({
                'success': True,
                'message': msg,
                'stopped_services': result.get('stopped_services', []),
            }), 200
        return jsonify({
            'success': False,
            'error': result.get('error', 'Failed to delete instance'),
            'errors': result.get('errors', []),
        }), 500
    except Exception as e:
        logger.error(f"API error (delete_instance): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/set-current/<instance_name>', methods=['POST'])
def api_set_current_instance(instance_name):
    """Set the current active instance."""
    try:
        instance = get_instance(instance_name)
        if instance:
            pm.current_instance = instance_name
            return jsonify({'success': True, 'message': f'Set current instance to {instance_name}'}), 200
        return jsonify({'success': False, 'error': 'Instance not found'}), 404
    except Exception as e:
        logger.error(f"API error (set_current_instance): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/start/<instance_name>', methods=['POST'])
def api_start_instance(instance_name):
    """Start an instance (launches auto-start services)."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404
        
        # Set as current instance
        pm.current_instance = instance_name
        
        # Start auto-start services
        started_services = []
        failed_services = []
        
        services_config = instance.get('services', {})
        for service_id, service_settings in services_config.items():
            if service_settings.get('enabled', False) and service_settings.get('auto_start', False):
                result = pm.start_service(service_id, instance_name)
                if result.get('success'):
                    started_services.append(service_id)
                    logger.info(f"Started auto-start service: {service_id}")
                else:
                    failed_services.append(service_id)
                    logger.warning(f"Failed to start: {service_id}")
        
        return jsonify({
            'success': True,
            'instance': instance_name,
            'started': started_services,
            'failed': failed_services,
            'message': f'Started {len(started_services)} auto-start services'
        }), 200
        
    except Exception as e:
        logger.error(f"API error (start_instance): {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/stop/<instance_name>', methods=['POST'])
def api_stop_instance(instance_name):
    """Stop an instance (stops all enabled services)."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404
        
        # Stop all enabled services
        stopped_services = []
        failed_services = []
        
        services_config = instance.get('services', {})
        for service_id, service_settings in services_config.items():
            if service_settings.get('enabled', False):
                result = pm.stop_service(service_id)
                if result.get('success'):
                    stopped_services.append(service_id)
                    logger.info(f"Stopped service: {service_id}")
                else:
                    failed_services.append(service_id)
                    logger.warning(f"Failed to stop: {service_id}")
        
        return jsonify({
            'success': True,
            'instance': instance_name,
            'stopped': stopped_services,
            'failed': failed_services,
            'message': f'Stopped {len(stopped_services)} services'
        }), 200
        
    except Exception as e:
        logger.error(f"API error (stop_instance): {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/get/<instance_name>', methods=['GET'])
def api_get_instance(instance_name):
    """Get instance data."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404
        
        instance['filename'] = instance_name
        return jsonify(instance), 200
        
    except Exception as e:
        logger.error(f"API error (get_instance): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API Endpoints - Instance-specific Service Endpoints
# ============================================================================

@app.route('/api/service/<instance_name>/<service_id>/start', methods=['POST'])
def api_service_start(instance_name, service_id):
    """Start a specific service in an instance."""
    try:
        pm.current_instance = instance_name
        result = pm.start_service(service_id, instance_name)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        logger.error(f"API error (service_start): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/service/<instance_name>/<service_id>/force-start', methods=['POST'])
def api_service_force_start(instance_name, service_id):
    """Force start a specific service: stop existing/port listeners and start fresh."""
    try:
        pm.current_instance = instance_name
        result = pm.start_service(service_id, instance_name, force=True)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        logger.error(f"API error (service_force_start): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/service/<instance_name>/<service_id>/stop', methods=['POST'])
def api_service_stop(instance_name, service_id):
    """Stop a specific service in an instance."""
    try:
        pm.current_instance = instance_name
        result = pm.stop_service(service_id)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        logger.error(f"API error (service_stop): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/service/<instance_name>/<service_id>/restart', methods=['POST'])
def api_service_restart(instance_name, service_id):
    """Restart a specific service in an instance."""
    try:
        pm.current_instance = instance_name
        result = pm.restart_service(service_id, instance_name)
        return jsonify(result), (200 if result.get('success') else 400)
    except Exception as e:
        logger.error(f"API error (service_restart): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/service/logs/<service_id>', methods=['GET'])
def api_service_logs(service_id):
    """Get recent logs for a specific service."""
    try:
        lines = request.args.get('lines', default=100, type=int)
        lines = max(1, min(lines, 1000))

        result = pm.get_logs(service_id, lines)
        return jsonify(result), (200 if result.get('success') else 404)
    except Exception as e:
        logger.error(f"API error (service_logs): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/<instance_name>/service/<service_id>/<config_key>', methods=['POST'])
def api_update_service_config(instance_name, service_id, config_key):
    """Update service configuration for an instance."""
    try:
        data = request.get_json()
        value = data.get('value')
        
        # Load instance
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404
        
        # Get service config for this instance
        if service_id not in instance.get('services', {}):
            return jsonify({'success': False, 'error': 'Service not in instance'}), 404
        
        # Update the service config value
        instance['services'][service_id][config_key] = value
        
        # Save the instance
        save_instance(instance_name, instance)
        
        return jsonify({'success': True, 'message': f'{config_key} updated to {value}'}), 200
    except Exception as e:
        logger.error(f"API error (update_service_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/<instance_name>/config/service/<service_id>', methods=['GET'])
def api_get_instance_service_runtime_config(instance_name, service_id):
    """Get runtime config.json for a service in a specific instance."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404

        if instance.get('services') and service_id not in instance.get('services', {}):
            return jsonify({'success': False, 'error': 'Service not in instance'}), 404

        config = load_instance_service_runtime_config(instance_name, service_id)
        if config is None:
            return jsonify({'success': False, 'error': 'Service config not found'}), 404

        config_file = get_instance_service_config_file(instance_name, service_id)
        config_path = str(config_file)
        return jsonify({'success': True, 'service_id': service_id, 'config': config, 'config_path': config_path}), 200
    except Exception as e:
        logger.error(f"API error (get_instance_service_runtime_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/instance/<instance_name>/config/service/<service_id>/save', methods=['POST'])
def api_save_instance_service_runtime_config(instance_name, service_id):
    """Save runtime config.json for a service in a specific instance."""
    try:
        instance = get_instance(instance_name)
        if not instance:
            return jsonify({'success': False, 'error': 'Instance not found'}), 404

        if instance.get('services') and service_id not in instance.get('services', {}):
            return jsonify({'success': False, 'error': 'Service not in instance'}), 404

        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400

        if save_instance_service_runtime_config(instance_name, service_id, data):
            return jsonify({'success': True, 'message': 'Config saved', 'service_id': service_id}), 200

        return jsonify({'success': False, 'error': 'Failed to save service config'}), 500
    except Exception as e:
        logger.error(f"API error (save_instance_service_runtime_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# API Endpoints - Config Management
# ============================================================================

@app.route('/api/config/get', methods=['GET'])
def api_get_config():
    """Get full configuration."""
    try:
        return jsonify({'success': True, 'config': pm.config}), 200
    except Exception as e:
        logger.error(f"API error (get_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/service/<service_id>', methods=['GET'])
def api_get_service_config(service_id):
    """Get service configuration."""
    try:
        config = load_service_config(service_id)
        if config:
            return jsonify({'success': True, 'config': config}), 200
        return jsonify({'success': False, 'error': 'Service not found'}), 404
    except Exception as e:
        logger.error(f"API error (get_service_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config/service/<service_id>/save', methods=['POST'])
def api_save_service_config(service_id):
    """Save service configuration."""
    try:
        data = request.get_json()
        if save_service_config(service_id, data):
            return jsonify({'success': True, 'message': 'Config saved'}), 200
        return jsonify({'success': False, 'error': 'Failed to save config'}), 500
    except Exception as e:
        logger.error(f"API error (save_service_config): {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    try:
        config = pm.config.get('manager', {})
        host = config.get('host', '0.0.0.0')
        port = config.get('port', 8000)
        debug = config.get('debug', False)
        
        logger.info(f"Starting Manager on {host}:{port} (debug={debug})")
        # use_reloader=False prevents crashes when instance JSON files are saved
        app.run(
            host=host, port=port, debug=debug,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}\n{traceback.format_exc()}")
        exit(1)
