"""Docker backend manager for controlling the faster-whisper container.

Design goals:
- Keep logic lightweight, no full docker compose parsing (would add complexity).
- Prefer Docker SDK for lifecycle of an existing container.
- If container does not exist, attempt to create it with sensible defaults (mirrors docker-compose.yml).
- Safe on systems without Docker: all methods fail gracefully and return status strings instead of raising.

Status values returned by this module (string):
  'running'  - container exists and running
  'stopped'  - container exists but not running
  'not_found' - container name not found
  'error'    - Docker not available / unexpected exception

GPU / advanced options:
Currently we do not automatically configure GPU. If GPU usage is required, start via docker compose manually first.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

try:
    import docker  # type: ignore
    from docker.errors import DockerException, NotFound  # type: ignore
except Exception:  # pragma: no cover - import guard
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore
    NotFound = Exception  # type: ignore


class DockerBackendManager:
    def __init__(
        self,
        container_name: str = "faster-whisper",
        image: str = "linuxserver/faster-whisper:gpu",
        port: int = 10300,
        default_env: Optional[Dict[str, str]] = None,
        auto_pull: bool = True,
    ):
        self.container_name = container_name
        self.image = image
        self.port = port
        self.default_env = default_env or {
            "PUID": "1000",
            "PGID": "1000",
            "TZ": "Etc/UTC",
            "WHISPER_MODEL": "turbo",
            "WHISPER_BEAM": "5",
            "WHISPER_LANG": "ru",
        }
        self.auto_pull = auto_pull
        self._client = None

    # --- internal helpers ---
    def _client_or_none(self):
        if docker is None:
            return None
        if self._client is None:
            try:
                self._client = docker.from_env()
            except Exception as e:  # pragma: no cover
                logger.debug(f"Failed to init docker client: {e}")
                return None
        return self._client

    def is_available(self) -> bool:
        cli = self._client_or_none()
        if cli is None:
            return False
        try:
            cli.ping()
            return True
        except Exception as e:
            logger.debug(f"Docker ping failed: {e}")
            return False

    # --- core container operations ---
    def _get_container(self):
        cli = self._client_or_none()
        if cli is None:
            return None
        try:
            return cli.containers.get(self.container_name)
        except NotFound:
            return None
        except Exception as e:  # pragma: no cover
            logger.debug(f"Error getting container {self.container_name}: {e}")
            return None

    def status(self) -> str:
        if not self.is_available():
            return "error"
        c = self._get_container()
        if c is None:
            return "not_found"
        c.reload()
        st = getattr(c, "status", "unknown")
        if st == "running":
            return "running"
        if st in ("exited", "created", "dead"):
            return "stopped"
        return st or "error"

    def start(self) -> str:
        """Start backend container. Creates container if missing.
        Returns resulting status string (see module docstring)."""
        if not self.is_available():
            logger.warning("Docker daemon not available – cannot start backend")
            return "error"
        cli = self._client_or_none()
        if cli is None:
            return "error"

        container = self._get_container()
        if container is not None:
            container.reload()
            if container.status == "running":
                return "running"
            try:
                container.start()
                container.reload()
                return "running" if container.status == "running" else "stopped"
            except DockerException as e:
                logger.error(f"Failed to start container: {e}")
                return "error"

        # Need to create new container
        try:
            if self.auto_pull:
                try:
                    logger.info(f"Pulling image {self.image} (if not present)...")
                    cli.images.pull(self.image)
                except Exception as e:
                    logger.debug(f"Image pull warning: {e}")
            logger.info("Creating faster-whisper container (first run)...")
            ports = {f"{self.port}/tcp": self.port}
            # Named volume for model cache similar to compose (auto created if absent)
            volumes = {"models_cache": {"bind": "/config", "mode": "rw"}}
            
            # GPU support for faster-whisper
            device_requests = [
                {
                    "driver": "nvidia",
                    "capabilities": [["gpu"]],
                    "count": -1  # -1 means all available GPUs
                }
            ]
            
            container = cli.containers.create(
                self.image,
                name=self.container_name,
                detach=True,
                environment=self.default_env,
                ports=ports,
                volumes=volumes,
                restart_policy={"Name": "unless-stopped"},
                device_requests=device_requests,
            )
            container.start()
            container.reload()
            logger.info("Container created and started.")
            return "running" if container.status == "running" else "stopped"
        except DockerException as e:
            logger.error(f"Failed to create/start container: {e}")
            return "error"

    def stop(self) -> str:
        if not self.is_available():
            return "error"
        container = self._get_container()
        if container is None:
            return "not_found"
        try:
            if container.status == "running":
                logger.info("Stopping faster-whisper container...")
                container.stop(timeout=10)
            container.reload()
            return "stopped" if container.status != "running" else "running"
        except DockerException as e:
            logger.error(f"Failed to stop container: {e}")
            return "error"

    def remove(self, force: bool = False) -> str:
        if not self.is_available():
            return "error"
        container = self._get_container()
        if container is None:
            return "not_found"
        try:
            name = container.name
            logger.info(f"Removing container {name} (force={force})...")
            container.remove(force=force)
            return "not_found"
        except DockerException as e:
            logger.error(f"Failed to remove container: {e}")
            return "error"

    def restart_with_model(self, model_alias: str) -> str:
        """Restart container with a new WHISPER_MODEL.
        This requires creating a new container because faster-whisper 
        loads model at startup via environment variable.
        
        Args:
            model_alias: Model alias (e.g., 'turbo') to use in container environment
        
        Returns resulting status string."""
        if not self.is_available():
            logger.warning("Docker daemon not available – cannot restart with new model")
            return "error"
        
        # Check if model is already set (avoid unnecessary restart)
        current_model = self.get_container_model_info()
        if current_model == model_alias:
            container = self._get_container()
            if container and container.status == "running":
                logger.info(f"Model alias '{model_alias}' already active in container")
                return "running"
        
        logger.info(f"Recreating container with model alias '{model_alias}'")
        
        # Update default environment with new model alias
        updated_env = self.default_env.copy()
        updated_env["WHISPER_MODEL"] = model_alias
        
        # Get existing container to preserve settings
        container = self._get_container()
        existing_ports = None
        
        if container is not None:
            try:
                # Preserve existing port configuration
                port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                existing_ports = {}
                for container_port, host_configs in port_bindings.items():
                    if host_configs:
                        existing_ports[container_port] = host_configs[0]['HostPort']
                
                # Stop and remove existing container
                if container.status == "running":
                    logger.info("Stopping existing container...")
                    container.stop(timeout=5)  # Reduced timeout for faster restart
                logger.info("Removing existing container...")
                container.remove()
                
            except DockerException as e:
                logger.error(f"Failed to stop/remove container for model switch: {e}")
                return "error"
        
        # Create and start new container with updated model
        cli = self._client_or_none()
        if cli is None:
            return "error"
        
        try:
            logger.info(f"Creating new container with model alias '{model_alias}'...")
            
            # Use preserved or default settings
            ports = existing_ports if existing_ports else {f"{self.port}/tcp": self.port}
            volumes = {"models_cache": {"bind": "/config", "mode": "rw"}}
            
            # GPU support for faster-whisper
            device_requests = [
                {
                    "driver": "nvidia",
                    "capabilities": [["gpu"]],
                    "count": -1  # -1 means all available GPUs
                }
            ]
            
            container = cli.containers.create(
                self.image,
                name=self.container_name,
                detach=True,
                environment=updated_env,
                ports=ports,
                volumes=volumes,
                restart_policy={"Name": "unless-stopped"},
                device_requests=device_requests,
            )
            container.start()
            container.reload()
            
            # Wait a moment for container to initialize
            import time
            time.sleep(2)
            
            logger.info(f"Container recreated successfully with model alias '{model_alias}'.")
            return "running" if container.status == "running" else "stopped"
            
        except DockerException as e:
            logger.error(f"Failed to create container with model alias '{model_alias}': {e}")
            return "error"

    # Convenience wrapper used by UI (returns tuple)
    def get_health_and_status(self, health_checker) -> tuple[str, bool]:
        """Return (status, health_ok) where:
        - status: container status string
        - health_ok: result of provided callable (bool) if running, else False
        """
        st = self.status()
        ok = False
        if st == "running":
            try:
                ok = bool(health_checker())
            except Exception:  # pragma: no cover
                ok = False
        return st, ok

    def get_container_model_info(self, engine=None) -> Optional[str]:
        """Get current model information from container environment or Wyoming engine.
        Returns model name if available, None if unavailable.
        
        Args:
            engine: Optional Wyoming engine to query for active model
        """
        # First try to get model from container environment variables (most reliable)
        try:
            container = self._get_container()
            if container:
                # Check if container is running
                container.reload()
                if container.status == 'running':
                    # Get environment variables from container
                    env_vars = container.attrs.get('Config', {}).get('Env', [])
                    for env in env_vars:
                        if env.startswith('WHISPER_MODEL='):
                            model_env = env.split('=', 1)[1]
                            return model_env
                            
        except Exception as e:
            logger.debug(f"Failed to get container model info: {e}")
        
        # Fallback to Wyoming engine if provided (less reliable for env changes)
        if engine and hasattr(engine, 'get_active_model'):
            try:
                active_model = engine.get_active_model()
                if active_model:
                    return active_model
            except Exception as e:
                logger.debug(f"Failed to get active model from Wyoming engine: {e}")
        
        return None

    def get_container_beam_info(self) -> Optional[int]:
        """Get current beam size from container environment variables.
        Returns beam size if available, None if unavailable.
        """
        try:
            container = self._get_container()
            if container:
                # Check if container is running
                container.reload()
                if container.status == 'running':
                    # Get environment variables from container
                    env_vars = container.attrs.get('Config', {}).get('Env', [])
                    for env in env_vars:
                        if env.startswith('WHISPER_BEAM='):
                            beam_env = env.split('=', 1)[1]
                            try:
                                return int(beam_env)
                            except ValueError:
                                return None
                            
        except Exception as e:
            logger.debug(f"Failed to get container beam info: {e}")
        
        return None

    def get_container_lang_info(self) -> Optional[str]:
        """Get current language from container environment variables.
        Returns language if available, None if unavailable.
        """
        try:
            container = self._get_container()
            if container:
                # Check if container is running
                container.reload()
                if container.status == 'running':
                    # Get environment variables from container
                    env_vars = container.attrs.get('Config', {}).get('Env', [])
                    for env in env_vars:
                        if env.startswith('WHISPER_LANG='):
                            lang_env = env.split('=', 1)[1]
                            return lang_env
                            
        except Exception as e:
            logger.debug(f"Failed to get container lang info: {e}")
        
        return None

    def restart_with_model_and_beam(self, model_alias: str, beam_size: int) -> str:
        """Restart container with a new WHISPER_MODEL and WHISPER_BEAM.
        This requires creating a new container because faster-whisper 
        loads configuration at startup via environment variables.
        
        Args:
            model_alias: Model alias (e.g., 'turbo') to use in container environment
            beam_size: Beam size value to use in container environment
        
        Returns resulting status string."""
        if not self.is_available():
            logger.warning("Docker daemon not available – cannot restart with new model and beam")
            return "error"
        
        # Check if both model and beam are already set (avoid unnecessary restart)
        current_model = self.get_container_model_info()
        current_beam = self.get_container_beam_info()
        if current_model == model_alias and current_beam == beam_size:
            container = self._get_container()
            if container and container.status == "running":
                logger.info(f"Model alias '{model_alias}' and beam size {beam_size} already active in container")
                return "running"
        
        logger.info(f"Recreating container with model alias '{model_alias}' and beam size {beam_size}")
        
        # Update default environment with new model alias and beam size
        updated_env = self.default_env.copy()
        updated_env["WHISPER_MODEL"] = model_alias
        updated_env["WHISPER_BEAM"] = str(beam_size)
        
        # Get existing container to preserve settings
        container = self._get_container()
        existing_ports = None
        
        if container is not None:
            try:
                # Preserve existing port configuration
                port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                existing_ports = {}
                for container_port, host_configs in port_bindings.items():
                    if host_configs:
                        existing_ports[container_port] = host_configs[0]['HostPort']
                
                # Stop and remove existing container
                if container.status == "running":
                    logger.info("Stopping existing container...")
                    container.stop(timeout=5)  # Reduced timeout for faster restart
                logger.info("Removing existing container...")
                container.remove()
                
            except DockerException as e:
                logger.error(f"Failed to stop/remove container for model/beam switch: {e}")
                return "error"
        
        # Create and start new container with updated model and beam
        cli = self._client_or_none()
        if cli is None:
            return "error"
        
        try:
            logger.info(f"Creating new container with model alias '{model_alias}' and beam size {beam_size}...")
            
            # Use preserved or default settings
            ports = existing_ports if existing_ports else {f"{self.port}/tcp": self.port}
            volumes = {"models_cache": {"bind": "/config", "mode": "rw"}}
            
            # GPU support for faster-whisper
            device_requests = [
                {
                    "driver": "nvidia",
                    "capabilities": [["gpu"]],
                    "count": -1  # -1 means all available GPUs
                }
            ]
            
            container = cli.containers.create(
                self.image,
                name=self.container_name,
                detach=True,
                environment=updated_env,
                ports=ports,
                volumes=volumes,
                restart_policy={"Name": "unless-stopped"},
                device_requests=device_requests,
            )
            container.start()
            container.reload()
            
            # Wait a moment for container to initialize
            import time
            time.sleep(2)
            
            logger.info(f"Container recreated successfully with model alias '{model_alias}' and beam size {beam_size}.")
            return "running" if container.status == "running" else "stopped"
            
        except DockerException as e:
            logger.error(f"Failed to create container with model alias '{model_alias}' and beam size {beam_size}: {e}")
            return "error"

    def restart_with_model_beam_and_lang(self, model_alias: str, beam_size: int, language: str) -> str:
        """Restart container with a new WHISPER_MODEL, WHISPER_BEAM, and WHISPER_LANG.
        This requires creating a new container because faster-whisper 
        loads configuration at startup via environment variables.
        
        Args:
            model_alias: Model alias (e.g., 'turbo') to use in container environment
            beam_size: Beam size value to use in container environment
            language: Language code (e.g., 'ru', 'en') to use in container environment
        
        Returns resulting status string."""
        if not self.is_available():
            logger.warning("Docker daemon not available – cannot restart with new model, beam, and language")
            return "error"
        
        # Check if all parameters are already set (avoid unnecessary restart)
        current_model = self.get_container_model_info()
        current_beam = self.get_container_beam_info()
        current_lang = self.get_container_lang_info()
        if current_model == model_alias and current_beam == beam_size and current_lang == language:
            container = self._get_container()
            if container and container.status == "running":
                logger.info(f"Model alias '{model_alias}', beam size {beam_size}, and language '{language}' already active in container")
                return "running"
        
        logger.info(f"Recreating container with model alias '{model_alias}', beam size {beam_size}, and language '{language}'")
        
        # Update default environment with new model alias, beam size, and language
        updated_env = self.default_env.copy()
        updated_env["WHISPER_MODEL"] = model_alias
        updated_env["WHISPER_BEAM"] = str(beam_size)
        updated_env["WHISPER_LANG"] = language
        
        # Get existing container to preserve settings
        container = self._get_container()
        existing_ports = None
        
        if container is not None:
            try:
                # Preserve existing port configuration
                port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                existing_ports = {}
                for container_port, host_configs in port_bindings.items():
                    if host_configs:
                        existing_ports[container_port] = host_configs[0]['HostPort']
                
                # Stop and remove existing container
                if container.status == "running":
                    logger.info("Stopping existing container...")
                    container.stop(timeout=5)  # Reduced timeout for faster restart
                logger.info("Removing existing container...")
                container.remove()
                
            except DockerException as e:
                logger.error(f"Failed to stop/remove container for model/beam/lang switch: {e}")
                return "error"
        
        # Create and start new container with updated model, beam, and language
        cli = self._client_or_none()
        if cli is None:
            return "error"
        
        try:
            logger.info(f"Creating new container with model alias '{model_alias}', beam size {beam_size}, and language '{language}'...")
            
            # Use preserved or default settings
            ports = existing_ports if existing_ports else {f"{self.port}/tcp": self.port}
            volumes = {"models_cache": {"bind": "/config", "mode": "rw"}}
            
            # GPU support for faster-whisper
            device_requests = [
                {
                    "driver": "nvidia",
                    "capabilities": [["gpu"]],
                    "count": -1  # -1 means all available GPUs
                }
            ]
            
            container = cli.containers.create(
                self.image,
                name=self.container_name,
                detach=True,
                environment=updated_env,
                ports=ports,
                volumes=volumes,
                restart_policy={"Name": "unless-stopped"},
                device_requests=device_requests,
            )
            container.start()
            container.reload()
            
            # Wait a moment for container to initialize
            import time
            time.sleep(2)
            
            logger.info(f"Container recreated successfully with model alias '{model_alias}', beam size {beam_size}, and language '{language}'.")
            return "running" if container.status == "running" else "stopped"
            
        except DockerException as e:
            logger.error(f"Failed to create container with model alias '{model_alias}', beam size {beam_size}, and language '{language}': {e}")
            return "error"
