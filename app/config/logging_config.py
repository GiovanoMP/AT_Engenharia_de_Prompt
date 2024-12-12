"""
Configurações de logging para o dashboard
"""

import logging
from pathlib import Path

def setup_logging():
    """Configura o sistema de logging da aplicação"""
    
    # Criar diretório de logs se não existir
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configuração básica
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'dashboard.log'),
            logging.StreamHandler()
        ]
    )
    
    # Logger específico para a aplicação
    logger = logging.getLogger('dashboard')
    logger.setLevel(logging.INFO)
    
    return logger

# Exceções customizadas
class DashboardException(Exception):
    """Exceção base para erros do dashboard"""
    pass

class DataLoadError(DashboardException):
    """Erro ao carregar dados"""
    pass

class ConfigError(DashboardException):
    """Erro de configuração"""
    pass
