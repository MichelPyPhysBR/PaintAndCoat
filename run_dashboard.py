#!/usr/bin/env python3
"""
Script de inicialização para o Dashboard Cientométrica
Verifica dependências e executa o dashboard automaticamente
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Verifica se o Streamlit está instalado"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False
    except FileNotFoundError:
        print("❌ Arquivo requirements_streamlit.txt não encontrado")
        return False

def run_dashboard():
    """Executa o dashboard"""
    print("🚀 Iniciando dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_cienciometrica.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard encerrado pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar dashboard: {e}")

def main():
    print("🔬 Dashboard Cientométrica - Tintas e Revestimentos Sustentáveis")
    print("=" * 65)
    
    # Verificar se o arquivo do dashboard existe
    if not Path("dashboard_cienciometrica.py").exists():
        print("❌ Arquivo dashboard_cienciometrica.py não encontrado!")
        print("📁 Certifique-se de que todos os arquivos estão no mesmo diretório")
        return
    
    # Verificar se o arquivo CSV existe
    if Path("scopus.csv").exists():
        print("✅ Arquivo scopus.csv encontrado - usando dados reais!")
        try:
            import pandas as pd
            df = pd.read_csv('scopus.csv', encoding='utf-8')
            print(f"📊 Dataset: {len(df)} artigos carregados")
        except Exception as e:
            print(f"⚠️  Problema com scopus.csv: {e}")
            print("💡 Execute 'python test_csv_loading.py' para diagnóstico")
    else:
        print("⚠️  Arquivo scopus.csv não encontrado")
        print("📄 O dashboard usará dados sintéticos para demonstração")
        print("💡 Coloque o arquivo scopus.csv na pasta para usar dados reais")
    
    # Verificar se Streamlit está instalado
    if not check_streamlit():
        print("📦 Streamlit não encontrado. Instalando dependências...")
        if not install_requirements():
            print("❌ Falha na instalação. Execute manualmente:")
            print("   pip install -r requirements_streamlit.txt")
            return
    
    print("✅ Dependências verificadas!")
    print("\n🌐 O dashboard será aberto em http://localhost:8501")
    print("💡 Para encerrar, pressione Ctrl+C no terminal")
    print("\n" + "-" * 50)
    
    # Executar dashboard
    run_dashboard()

if __name__ == "__main__":
    main()