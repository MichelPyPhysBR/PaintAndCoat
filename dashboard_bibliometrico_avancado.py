import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import re
import warnings
import itertools
warnings.filterwarnings('ignore')

# Tentar importar bibliotecas opcionais
try:
    from scipy.stats import powerlaw
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False
    st.warning("⚠️ Algumas bibliotecas avançadas não estão disponíveis. Instale com: pip install scipy scikit-learn")

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Configurações da página
st.set_page_config(
    page_title="Dashboard Bibliométrico Avançado - Tintas Sustentáveis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1e40af;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #0ea5e9;
        margin: 1rem 0;
    }
    .quadrant-box {
        background-color: #fefce8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #eab308;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados reais do Scopus"""
    try:
        df = pd.read_csv('scopus.csv', encoding='utf-8')
        
        # Renomear colunas
        column_mapping = {
            'Cited by': 'Citations',
            'Author Keywords': 'Author_Keywords',
            'Index Keywords': 'Index_Keywords',
            'Source title': 'Journal',
            'Document Type': 'Document_Type'
        }
        df = df.rename(columns=column_mapping)
        
        # Processamento
        df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce').fillna(0).astype(int)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Author_Keywords'] = df['Author_Keywords'].fillna('')
        df['Index_Keywords'] = df['Index_Keywords'].fillna('')
        df['Journal'] = df['Journal'].fillna('Unknown')
        df['Authors'] = df['Authors'].fillna('')
        df['Affiliations'] = df['Affiliations'].fillna('') if 'Affiliations' in df.columns else ''
        
        df = df[df['Year'] > 1900]
        st.success(f"✅ Dados carregados: {len(df)} artigos do Scopus")
        
    except FileNotFoundError:
        st.error("❌ Arquivo scopus.csv não encontrado! Usando dados sintéticos.")
        df = create_synthetic_data()
    except Exception as e:
        st.error(f"❌ Erro: {e}. Usando dados sintéticos.")
        df = create_synthetic_data()
    
    return df

def create_synthetic_data():
    """Cria dados sintéticos para demonstração"""
    np.random.seed(42)
    n_papers = 800
    
    years = np.random.choice(range(2015, 2025), n_papers, 
                            p=[0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.15, 0.1, 0.05, 0.02])
    citations = np.random.exponential(3, n_papers).astype(int)
    
    # Keywords temáticas mais realistas
    themes = {
        'sustainability': ['sustainable coating', 'eco-friendly', 'green chemistry', 'bio-based', 'renewable'],
        'nanotechnology': ['nanoparticles', 'nanomaterials', 'nanocomposite', 'graphene', 'carbon nanotubes'],
        'functionality': ['self-healing', 'antimicrobial', 'anti-corrosion', 'photocatalytic', 'smart coating'],
        'materials': ['polymer', 'metal oxide', 'titanium dioxide', 'zinc oxide', 'silica'],
        'applications': ['automotive', 'marine', 'construction', 'aerospace', 'biomedical']
    }
    
    authors_pool = [
        "Zhang, L.", "Smith, J.", "Wang, Y.", "Brown, M.", "Li, X.",
        "Johnson, R.", "Chen, W.", "Davis, S.", "Liu, H.", "Wilson, K.",
        "Garcia, A.", "Martinez, C.", "Anderson, P.", "Taylor, D.", "Thomas, J."
    ]
    
    journals_pool = [
        "Progress in Organic Coatings", "Surface and Coatings Technology", 
        "Journal of Coatings Technology", "Applied Surface Science",
        "Green Chemistry", "Journal of Cleaner Production", "ACS Applied Materials"
    ]
    
    data = []
    for i in range(n_papers):
        # Selecionar tema principal
        main_theme = np.random.choice(list(themes.keys()))
        theme_keywords = themes[main_theme]
        
        # Adicionar keywords de outros temas (co-ocorrência)
        all_keywords = theme_keywords.copy()
        for other_theme in themes.keys():
            if other_theme != main_theme and np.random.random() < 0.3:
                all_keywords.extend(np.random.choice(themes[other_theme], 1))
        
        keywords_str = "; ".join(np.random.choice(all_keywords, 
                                                min(len(all_keywords), np.random.randint(3, 7)), 
                                                replace=False))
        
        # Autores
        n_authors = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        authors = np.random.choice(authors_pool, n_authors, replace=False)
        authors_str = "; ".join(authors)
        
        data.append({
            'Title': f"Sustainable Coating Research {i+1}: {theme_keywords[0].title()}",
            'Authors': authors_str,
            'Year': years[i],
            'Citations': citations[i],
            'Journal': np.random.choice(journals_pool),
            'Author_Keywords': keywords_str,
            'Index_Keywords': keywords_str,
            'Affiliations': f"University {chr(65 + i % 26)}, {np.random.choice(['USA', 'China', 'Germany', 'Japan', 'UK'])}"
        })
    
    return pd.DataFrame(data)

@st.cache_data
def extract_keywords_with_cooccurrence(df):
    """Extrai palavras-chave e calcula co-ocorrência"""
    all_keywords = []
    keyword_pairs = []
    
    for idx, row in df.iterrows():
        paper_keywords = []
        
        # Combinar author e index keywords
        for kw_col in ['Author_Keywords', 'Index_Keywords']:
            if pd.notna(row[kw_col]) and str(row[kw_col]).strip():
                kws = [kw.strip().lower() for kw in str(row[kw_col]).split(';')]
                paper_keywords.extend(kws)
        
        # Limpar e deduplicate
        paper_keywords = [kw for kw in set(paper_keywords) if kw and len(kw) > 2]
        
        # Adicionar keywords individuais
        for kw in paper_keywords:
            all_keywords.append({
                'Paper_ID': idx,
                'Keyword': kw,
                'Year': row['Year'],
                'Citations': row['Citations']
            })
        
        # Calcular pares (co-ocorrência)
        for pair in itertools.combinations(paper_keywords, 2):
            keyword_pairs.append({
                'Keyword1': pair[0],
                'Keyword2': pair[1],
                'Year': row['Year'],
                'Paper_ID': idx
            })
    
    keywords_df = pd.DataFrame(all_keywords)
    cooccurrence_df = pd.DataFrame(keyword_pairs)
    
    return keywords_df, cooccurrence_df

def calculate_h_index(citations_sorted):
    """Calcula índice H"""
    h_index = 0
    for i, citations in enumerate(citations_sorted, 1):
        if citations >= i:
            h_index = i
        else:
            break
    return h_index

def analyze_lotka_law(author_stats):
    """Analisa Lei de Lotka (distribuição de produtividade)"""
    if len(author_stats) == 0:
        return None, None, None
        
    productivity = author_stats['Papers_Count'].values
    unique_counts = np.unique(productivity)
    freq = [np.sum(productivity == count) for count in unique_counts]
    
    # Ajuste da Lei de Lotka: f(x) = C / x^α
    # Log transform: log(f) = log(C) - α * log(x)
    try:
        log_x = np.log(unique_counts)
        log_y = np.log(freq)
        
        # Regressão linear no espaço log
        coef = np.polyfit(log_x, log_y, 1)
        alpha = -coef[0]  # Expoente de Lotka
        C = np.exp(coef[1])  # Constante
        
        return unique_counts, freq, alpha
    except:
        return unique_counts, freq, None

def create_thematic_evolution(keywords_df):
    """Cria análise de evolução temática"""
    if len(keywords_df) == 0:
        return pd.DataFrame()
        
    # Agrupar por palavra-chave e ano
    evolution = keywords_df.groupby(['Keyword', 'Year']).agg({
        'Paper_ID': 'count',
        'Citations': 'sum'
    }).reset_index()
    
    evolution.columns = ['Keyword', 'Year', 'Frequency', 'Citations']
    
    # Calcular tendência (slope)
    trend_data = []
    for keyword in evolution['Keyword'].unique():
        kw_data = evolution[evolution['Keyword'] == keyword]
        if len(kw_data) >= 3:  # Mínimo 3 anos
            try:
                slope = np.polyfit(kw_data['Year'], kw_data['Frequency'], 1)[0]
                total_freq = kw_data['Frequency'].sum()
                trend_data.append({
                    'Keyword': keyword,
                    'Trend': slope,
                    'Total_Frequency': total_freq,
                    'Total_Citations': kw_data['Citations'].sum()
                })
            except:
                continue
    
    return pd.DataFrame(trend_data)

def create_thematic_map(keywords_df, cooccurrence_df):
    """Cria mapa temático com quadrantes"""
    if len(keywords_df) == 0:
        return pd.DataFrame()
        
    # Calcular centralidade (co-ocorrência) e densidade
    keyword_stats = keywords_df.groupby('Keyword').agg({
        'Paper_ID': 'count',
        'Citations': 'sum'
    }).reset_index()
    keyword_stats.columns = ['Keyword', 'Frequency', 'Citations']
    
    # Centralidade: quantas vezes aparece com outras palavras
    centrality = {}
    for keyword in keyword_stats['Keyword']:
        cooc_count = len(cooccurrence_df[
            (cooccurrence_df['Keyword1'] == keyword) | 
            (cooccurrence_df['Keyword2'] == keyword)
        ])
        centrality[keyword] = cooc_count
    
    keyword_stats['Centrality'] = keyword_stats['Keyword'].map(centrality).fillna(0)
    keyword_stats['Density'] = keyword_stats['Citations'] / keyword_stats['Frequency']
    
    # Normalizar para classificação em quadrantes
    if len(keyword_stats) > 0 and keyword_stats['Centrality'].std() > 0:
        keyword_stats['Centrality_Norm'] = (keyword_stats['Centrality'] - keyword_stats['Centrality'].mean()) / keyword_stats['Centrality'].std()
        keyword_stats['Density_Norm'] = (keyword_stats['Density'] - keyword_stats['Density'].mean()) / keyword_stats['Density'].std()
        
        # Classificar em quadrantes
        def classify_quadrant(row):
            if row['Centrality_Norm'] > 0 and row['Density_Norm'] > 0:
                return 'Motor Themes'
            elif row['Centrality_Norm'] > 0 and row['Density_Norm'] <= 0:
                return 'Basic Themes'
            elif row['Centrality_Norm'] <= 0 and row['Density_Norm'] > 0:
                return 'Niche Themes'
            else:
                return 'Emerging/Declining'
        
        keyword_stats['Quadrant'] = keyword_stats.apply(classify_quadrant, axis=1)
    
    return keyword_stats

def create_collaboration_network(df):
    """Cria rede de colaboração entre autores"""
    author_pairs = []
    
    for _, row in df.iterrows():
        if pd.notna(row['Authors']):
            authors = [a.strip() for a in str(row['Authors']).split(';')]
            if len(authors) > 1:
                for pair in itertools.combinations(authors, 2):
                    author_pairs.append(pair)
    
    # Contar colaborações
    collab_counts = Counter(author_pairs)
    
    # Criar rede
    G = nx.Graph()
    for (author1, author2), weight in collab_counts.items():
        if weight >= 2:  # Filtrar colaborações mínimas
            G.add_edge(author1, author2, weight=weight)
    
    return G

def safe_divide(a, b):
    """Divisão segura para evitar divisão por zero"""
    return a / b if b != 0 else 0

def main():
    st.markdown('<div class="main-header">🔬 Dashboard Bibliométrico Avançado<br>Tintas e Revestimentos Sustentáveis</div>', unsafe_allow_html=True)
    
    # Carregamento dos dados
    with st.spinner("Carregando dados e preparando análises avançadas..."):
        df = load_and_process_data()
        keywords_df, cooccurrence_df = extract_keywords_with_cooccurrence(df)
    
    # Calcular métricas principais uma vez para uso global
    h_index = calculate_h_index(df['Citations'].sort_values(ascending=False))
    
    # Calcular taxa de colaboração de forma segura
    multi_author_papers = 0
    for authors_str in df['Authors'].dropna():
        authors = [a.strip() for a in str(authors_str).split(';')]
        if len(authors) > 1:
            multi_author_papers += 1
    
    collaboration_rate = safe_divide(multi_author_papers, len(df))
    
    # Sidebar
    st.sidebar.header("📋 Análises Bibliométricas")
    st.sidebar.markdown(f"""
    **Dataset:** {len(df)} artigos
    **Período:** {df['Year'].min()}-{df['Year'].max()}
    **Keywords:** {len(keywords_df['Keyword'].unique()) if len(keywords_df) > 0 else 0} únicas
    **Co-ocorrências:** {len(cooccurrence_df)} pares
    """)
    
    # Seleção de análises
    analysis_option = st.sidebar.selectbox(
        "Escolha a análise:",
        [
            "1. Dados Gerais",
            "2. Evolução Temática", 
            "3. Mapa Temático",
            "4. Trend Topics",
            "5. Co-ocorrência de Keywords",
            "6. Redes de Colaboração",
            "7. Análise de Co-citação",
            "8. Top Papers (Citações)",
            "9. Lei de Lotka (Autores)"
        ]
    )
    
    # ===============================
    # 1. DADOS GERAIS
    # ===============================
    if "1. Dados Gerais" in analysis_option:
        st.markdown('<div class="section-header">📊 1. Dados Gerais & Análise Descritiva</div>', unsafe_allow_html=True)
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Artigos", f"{len(df):,}")
        with col2:
            st.metric("Total Citações", f"{df['Citations'].sum():,}")
        with col3:
            unique_authors = len(set([a.strip() for authors in df['Authors'].dropna() for a in str(authors).split(';') if a.strip()]))
            st.metric("Autores Únicos", f"{unique_authors:,}")
        with col4:
            st.metric("Índice H", f"{h_index}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Taxa Colaboração", f"{collaboration_rate*100:.1f}%")
        with col2:
            st.metric("Período", f"{df['Year'].min()}-{df['Year'].max()}")
        with col3:
            st.metric("Revistas Únicas", f"{df['Journal'].nunique()}")
        with col4:
            avg_citations = df['Citations'].mean()
            st.metric("Média Citações", f"{avg_citations:.1f}")
        
        st.markdown("""
        <div class="explanation-box">
        <h4>📈 Por que usar Dados Gerais?</h4>
        <ul>
            <li><strong>Panorama básico:</strong> Número total de documentos, autores e citações</li>
            <li><strong>Evolução temporal:</strong> Mostra crescimento da produção científica anual</li>
            <li><strong>Principais fontes:</strong> Identifica jornais e países mais produtivos</li>
            <li><strong>Base quantitativa:</strong> Fundamenta introdução e metodologia com estatísticas</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Produção anual
        yearly_prod = df['Year'].value_counts().sort_index()
        fig_yearly = px.line(x=yearly_prod.index, y=yearly_prod.values, 
                           title="Produção Científica Anual", markers=True)
        fig_yearly.update_layout(xaxis_title="Ano", yaxis_title="Número de Artigos")
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        # Top journals
        top_journals = df['Journal'].value_counts().head(10)
        fig_journals = px.bar(y=top_journals.index, x=top_journals.values, orientation='h',
                            title="Top 10 Revistas Mais Produtivas")
        fig_journals.update_layout(yaxis_title="Revista", xaxis_title="Número de Artigos")
        st.plotly_chart(fig_journals, use_container_width=True)
        
        # Explicação dos resultados
        st.markdown("""
        <div class="explanation-box">
        <h4>🔍 Como Interpretar os Resultados:</h4>
        <ul>
            <li><strong>Crescimento anual:</strong> Tendência ascendente indica área em expansão e oportunidades de financiamento</li>
            <li><strong>Top revistas:</strong> Identifica onde publicar e quais acompanhar para estar atualizado</li>
            <li><strong>Picos de produção:</strong> Podem coincidir com marcos regulatórios (ex: normas ambientais)</li>
            <li><strong>Distribuição geográfica:</strong> Revela centros de expertise para parcerias</li>
        </ul>
        <p><strong>💡 Ação recomendada:</strong> Foque nas revistas top 5 para submissões e monitore tendências anuais para timing de projetos.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===============================
    # 2. EVOLUÇÃO TEMÁTICA
    # ===============================
    elif "2. Evolução Temática" in analysis_option:
        st.markdown('<div class="section-header">🌱 2. Evolução Temática</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Evolução Temática?</h4>
        <ul>
            <li><strong>Emergência de temas:</strong> Identifica como temas amadurecem ou declinam</li>
            <li><strong>Novos materiais:</strong> Rastreia óxidos metálicos nanométricos, autorreparação</li>
            <li><strong>Lacunas de pesquisa:</strong> Fornece insights para áreas promissoras</li>
            <li><strong>Antecipação:</strong> Permite prever tendências futuras</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        evolution_data = create_thematic_evolution(keywords_df)
        
        if len(evolution_data) > 0:
            # Top trending keywords
            trending_up = evolution_data.nlargest(10, 'Trend')
            trending_down = evolution_data.nsmallest(10, 'Trend')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🚀 Temas Emergentes (Trend ↗️)")
                if len(trending_up) > 0:
                    fig_up = px.bar(trending_up, x='Trend', y='Keyword', orientation='h',
                                  color='Total_Citations', title="Keywords com Maior Crescimento")
                    st.plotly_chart(fig_up, use_container_width=True)
            
            with col2:
                st.subheader("📉 Temas em Declínio (Trend ↘️)")
                if len(trending_down) > 0:
                    fig_down = px.bar(trending_down, x='Trend', y='Keyword', orientation='h',
                                    color='Total_Citations', title="Keywords em Declínio")
                    st.plotly_chart(fig_down, use_container_width=True)
            
            # Heatmap de evolução temporal
            st.subheader("🔥 Mapa de Calor - Evolução Temporal")
            
            # Preparar dados para heatmap
            keywords_df_year = keywords_df.groupby(['Keyword', 'Year']).size().reset_index(name='Count')
            top_keywords = keywords_df['Keyword'].value_counts().head(20).index
            
            heatmap_data = keywords_df_year[keywords_df_year['Keyword'].isin(top_keywords)]
            if len(heatmap_data) > 0:
                pivot_data = heatmap_data.pivot(index='Keyword', columns='Year', values='Count').fillna(0)
                
                fig_heatmap = px.imshow(pivot_data, 
                                      title="Evolução Temporal das Top 20 Keywords",
                                      color_continuous_scale='Viridis')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Explicação dos resultados
            st.markdown("""
            <div class="explanation-box">
            <h4>🔍 Como Interpretar a Evolução Temática:</h4>
            <ul>
                <li><strong>Temas Emergentes (Trend ↗️):</strong> Áreas com crescimento acelerado - oportunidades de inovação</li>
                <li><strong>Temas em Declínio (Trend ↘️):</strong> Podem indicar saturação ou mudança de paradigma</li>
                <li><strong>Heatmap temporal:</strong> Cores quentes = alta atividade, cores frias = baixa atividade</li>
                <li><strong>Padrões sazonais:</strong> Alguns temas podem ter ciclos relacionados a regulamentações</li>
            </ul>
            <p><strong>💡 Para Tintas Sustentáveis:</strong></p>
            <ul>
                <li><strong>Se "bio-based" está emergente:</strong> Invista em polímeros naturais</li>
                <li><strong>Se "nanoparticles" declina:</strong> Foque em alternativas sustentáveis</li>
                <li><strong>Timing estratégico:</strong> Entre em temas emergentes antes do pico de competição</li>
            </ul>
            <p><strong>🎯 Ação recomendada:</strong> Monitore temas com trend positivo consistente por 3+ anos para investimento em P&D.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados insuficientes para análise de evolução temática")
    
    # ===============================
    # 3. MAPA TEMÁTICO
    # ===============================
    elif "3. Mapa Temático" in analysis_option:
        st.markdown('<div class="section-header">🗺️ 3. Mapa Temático</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Mapa Temático?</h4>
        <ul>
            <li><strong>Classificação por quadrantes:</strong> Motor, básico, emergente/periférico</li>
            <li><strong>Priorização:</strong> Diferencia tópicos centrais vs exploratórios</li>
            <li><strong>Áreas consolidadas:</strong> Ex: auto-limpeza vs sensores fotocatalíticos</li>
            <li><strong>Decisões estratégicas:</strong> Facilita desenvolvimento e financiamento</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        thematic_map = create_thematic_map(keywords_df, cooccurrence_df)
        
        if len(thematic_map) > 0 and 'Quadrant' in thematic_map.columns:
            # Scatter plot dos quadrantes
            fig_map = px.scatter(thematic_map, x='Centrality_Norm', y='Density_Norm',
                               size='Frequency', color='Quadrant', hover_name='Keyword',
                               title="Mapa Temático - Classificação por Quadrantes")
            
            # Adicionar linhas dos quadrantes
            fig_map.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_map.add_vline(x=0, line_dash="dash", line_color="gray")
            
            # Labels dos quadrantes
            fig_map.add_annotation(x=1.5, y=1.5, text="MOTOR THEMES<br>(High Centrality & Density)", 
                                 showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            fig_map.add_annotation(x=1.5, y=-1.5, text="BASIC THEMES<br>(High Centrality, Low Density)", 
                                 showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            fig_map.add_annotation(x=-1.5, y=1.5, text="NICHE THEMES<br>(Low Centrality, High Density)", 
                                 showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            fig_map.add_annotation(x=-1.5, y=-1.5, text="EMERGING/DECLINING<br>(Low Centrality & Density)", 
                                 showarrow=False, bgcolor="rgba(255,255,255,0.8)")
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Resumo por quadrante
            quadrant_summary = thematic_map.groupby('Quadrant').agg({
                'Keyword': 'count',
                'Frequency': 'sum',
                'Citations': 'sum'
            }).round(2)
            
            st.subheader("📊 Resumo por Quadrante")
            for quadrant in quadrant_summary.index:
                keywords_in_quad = thematic_map[thematic_map['Quadrant'] == quadrant]['Keyword'].tolist()
                st.markdown(f"""
                <div class="quadrant-box">
                <h4>{quadrant}</h4>
                <p><strong>Keywords ({len(keywords_in_quad)}):</strong> {', '.join(keywords_in_quad[:5])}{'...' if len(keywords_in_quad) > 5 else ''}</p>
                <p><strong>Total Citations:</strong> {quadrant_summary.loc[quadrant, 'Citations']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Explicação estratégica dos quadrantes
            st.markdown("""
            <div class="explanation-box">
            <h4>🔍 Como Interpretar o Mapa Temático:</h4>
            
            <p><strong>🚀 MOTOR THEMES (Alta Centralidade + Alta Densidade):</strong></p>
            <ul>
                <li><strong>O que são:</strong> Temas centrais e bem desenvolvidos da área</li>
                <li><strong>Estratégia:</strong> Mantenha investimento para liderar mercado</li>
                <li><strong>Exemplo:</strong> Se "coating performance" está aqui, é área consolidada para inovar</li>
            </ul>
            
            <p><strong>📚 BASIC THEMES (Alta Centralidade + Baixa Densidade):</strong></p>
            <ul>
                <li><strong>O que são:</strong> Temas fundamentais mas pouco explorados</li>
                <li><strong>Estratégia:</strong> Oportunidade de especialização e diferenciação</li>
                <li><strong>Exemplo:</strong> "sustainability" pode estar aqui - conceito amplo, pouco específico</li>
            </ul>
            
            <p><strong>🎯 NICHE THEMES (Baixa Centralidade + Alta Densidade):</strong></p>
            <ul>
                <li><strong>O que são:</strong> Especializações técnicas específicas</li>
                <li><strong>Estratégia:</strong> Nichos rentáveis para empresas especializadas</li>
                <li><strong>Exemplo:</strong> "marine coatings" - aplicação específica mas bem desenvolvida</li>
            </ul>
            
            <p><strong>🌱 EMERGING/DECLINING (Baixa Centralidade + Baixa Densidade):</strong></p>
            <ul>
                <li><strong>O que são:</strong> Temas nascentes ou em declínio</li>
                <li><strong>Estratégia:</strong> Monitore emergentes, abandone os em declínio</li>
                <li><strong>Exemplo:</strong> "smart coatings" pode estar emergindo</li>
            </ul>
            
            <p><strong>🎯 Ação Estratégica:</strong></p>
            <ul>
                <li><strong>70% recursos:</strong> Motor Themes (liderança)</li>
                <li><strong>20% recursos:</strong> Basic Themes (diferenciação)</li>
                <li><strong>10% recursos:</strong> Emerging Themes (apostas futuras)</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados insuficientes para mapa temático")
    
    # ===============================
    # 4. TREND TOPICS
    # ===============================
    elif "4. Trend Topics" in analysis_option:
        st.markdown('<div class="section-header">📈 4. Análise de Tendências (Trend Topics)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Trend Topics?</h4>
        <ul>
            <li><strong>Frequência temporal:</strong> Revela picos de interesse (ex: "grafeno em tintas")</li>
            <li><strong>Novos termos:</strong> Identifica recém-chegados ("produção em escala industrial")</li>
            <li><strong>Maturidade tecnológica:</strong> Compreende ciclo de hype</li>
            <li><strong>Timing de pesquisa:</strong> Orienta quando entrar em novos tópicos</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Análise de trending topics
        if len(keywords_df) > 0:
            # Top keywords por ano
            yearly_keywords = keywords_df.groupby(['Year', 'Keyword']).size().reset_index(name='Count')
            
            # Selecionar top keywords
            top_kw = keywords_df['Keyword'].value_counts().head(15).index
            trend_data = yearly_keywords[yearly_keywords['Keyword'].isin(top_kw)]
            
            # Heatmap de tendências
            if len(trend_data) > 0:
                pivot_trend = trend_data.pivot(index='Keyword', columns='Year', values='Count').fillna(0)
                
                fig_trend = px.imshow(pivot_trend, 
                                    title="Heatmap de Tendências - Top 15 Keywords",
                                    color_continuous_scale='RdYlBu_r')
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Linha temporal para keywords selecionadas
            st.subheader("📊 Evolução Temporal de Keywords Específicas")
            
            selected_keywords = st.multiselect(
                "Selecione até 5 keywords para análise temporal:",
                options=list(top_kw),
                default=list(top_kw[:3])
            )
            
            if selected_keywords:
                filter_data = trend_data[trend_data['Keyword'].isin(selected_keywords)]
                fig_lines = px.line(filter_data, x='Year', y='Count', color='Keyword',
                                  title="Evolução Temporal das Keywords Selecionadas",
                                  markers=True)
                st.plotly_chart(fig_lines, use_container_width=True)
            
            # Detecção de keywords emergentes
            st.subheader("🚀 Detecção de Keywords Emergentes")
            
            # Keywords que apareceram recentemente (últimos 3 anos)
            recent_years = df['Year'].max() - 2
            recent_keywords = keywords_df[keywords_df['Year'] >= recent_years]['Keyword'].value_counts()
            all_time_keywords = keywords_df['Keyword'].value_counts()
            
            # Calcular ratio de frequência recente vs total
            emerging_score = {}
            for kw in recent_keywords.index:
                if all_time_keywords[kw] >= 3:  # Mínimo de ocorrências
                    ratio = recent_keywords[kw] / all_time_keywords[kw]
                    emerging_score[kw] = ratio
            
            # Top emerging keywords
            if emerging_score:
                emerging_df = pd.DataFrame(list(emerging_score.items()), 
                                         columns=['Keyword', 'Emerging_Score'])
                emerging_df = emerging_df.sort_values('Emerging_Score', ascending=False).head(10)
                
                fig_emerging = px.bar(emerging_df, x='Emerging_Score', y='Keyword', 
                                    orientation='h',
                                    title="Top 10 Keywords Emergentes (Ratio Recente/Total)")
                st.plotly_chart(fig_emerging, use_container_width=True)
            
            # Explicação dos trends
            st.markdown("""
            <div class="explanation-box">
            <h4>🔍 Como Interpretar Trend Topics:</h4>
            
            <p><strong>📈 Heatmap de Tendências:</strong></p>
            <ul>
                <li><strong>Cores quentes (vermelho/amarelo):</strong> Períodos de alta atividade</li>
                <li><strong>Cores frias (azul):</strong> Baixa atividade ou temas dormentes</li>
                <li><strong>Padrões horizontais:</strong> Temas consistentes ao longo do tempo</li>
                <li><strong>Padrões verticais:</strong> Anos de alta atividade geral</li>
            </ul>
            
            <p><strong>📊 Evolução Temporal:</strong></p>
            <ul>
                <li><strong>Curvas ascendentes:</strong> Temas ganhando momentum</li>
                <li><strong>Picos isolados:</strong> Podem indicar eventos específicos (regulamentações)</li>
                <li><strong>Declínios graduais:</strong> Saturação ou mudança tecnológica</li>
                <li><strong>Ressurgências:</strong> Temas "voltando à moda" com novas abordagens</li>
            </ul>
            
            <p><strong>🚀 Keywords Emergentes (Emerging Score):</strong></p>
            <ul>
                <li><strong>Score > 0.7:</strong> Altamente emergente - investimento prioritário</li>
                <li><strong>Score 0.4-0.7:</strong> Crescimento moderado - monitorar</li>
                <li><strong>Score < 0.4:</strong> Estável ou maduro</li>
            </ul>
            
            <p><strong>💡 Para Tintas Sustentáveis - Ações por Trend:</strong></p>
            <ul>
                <li><strong>Se "circular economy" está emergente:</strong> Desenvolva programas de reciclagem</li>
                <li><strong>Se "VOC-free" está crescendo:</strong> Invista em formulações água-base</li>
                <li><strong>Se "bio-based polymers" tem pico:</strong> Parcerias com fornecedores naturais</li>
                <li><strong>Se "nanotechnology" declina:</strong> Foque em nano-segurança e sustentabilidade</li>
            </ul>
            
            <p><strong>🎯 Timing Estratégico:</strong> Entre em temas emergentes 1-2 anos antes do pico para maximizar vantagem competitiva.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados insuficientes para análise de trends")
    
    # ===============================
    # 5. CO-OCORRÊNCIA DE KEYWORDS
    # ===============================
    elif "5. Co-ocorrência de Keywords" in analysis_option:
        st.markdown('<div class="section-header">🔗 5. Análise de Co-ocorrência de Keywords</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Co-ocorrência?</h4>
        <ul>
            <li><strong>Relacionamentos:</strong> Desvela conexões entre conceitos</li>
            <li><strong>Exemplo:</strong> "auto-reparação" vs "liberação controlada"</li>
            <li><strong>Clusters temáticos:</strong> Mapeia grupos de temas relacionados</li>
            <li><strong>Parcerias:</strong> Orienta revisões e colaborações multidisciplinares</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if len(cooccurrence_df) > 0:
            # Calcular matriz de co-ocorrência
            cooc_counts = cooccurrence_df.groupby(['Keyword1', 'Keyword2']).size().reset_index(name='Count')
            
            # Filtrar por frequência mínima
            min_cooc = st.slider("Frequência mínima de co-ocorrência:", 1, 10, 3)
            cooc_filtered = cooc_counts[cooc_counts['Count'] >= min_cooc]
            
            if len(cooc_filtered) > 0:
                # Criar rede de co-ocorrência
                G = nx.Graph()
                for _, row in cooc_filtered.iterrows():
                    G.add_edge(row['Keyword1'], row['Keyword2'], weight=row['Count'])
                
                st.write(f"🔗 Rede de co-ocorrência: {G.number_of_nodes()} keywords, {G.number_of_edges()} conexões")
                
                # Calcular métricas de rede
                if G.number_of_nodes() > 0:
                    centrality = nx.degree_centrality(G)
                    betweenness = nx.betweenness_centrality(G)
                    
                    # Top keywords por centralidade
                    central_df = pd.DataFrame([
                        {'Keyword': k, 'Degree_Centrality': v, 'Betweenness': betweenness[k]}
                        for k, v in centrality.items()
                    ]).sort_values('Degree_Centrality', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 Top Keywords por Centralidade")
                        if len(central_df) > 0:
                            fig_central = px.bar(central_df.head(10), x='Degree_Centrality', y='Keyword',
                                               orientation='h', title="Degree Centrality")
                            st.plotly_chart(fig_central, use_container_width=True)
                    
                    with col2:
                        st.subheader("🌉 Top Keywords por Betweenness")
                        if len(central_df) > 0:
                            fig_between = px.bar(central_df.head(10), x='Betweenness', y='Keyword',
                                               orientation='h', title="Betweenness Centrality")
                            st.plotly_chart(fig_between, use_container_width=True)
                
                # Matriz de co-ocorrência (heatmap)
                st.subheader("🔥 Matriz de Co-ocorrência")
                
                # Criar matriz pivot
                if len(keywords_df) > 0:
                    top_keywords = keywords_df['Keyword'].value_counts().head(20).index
                    matrix_data = []
                    
                    for kw1 in top_keywords:
                        row = []
                        for kw2 in top_keywords:
                            if kw1 == kw2:
                                row.append(0)
                            else:
                                cooc = cooccurrence_df[
                                    ((cooccurrence_df['Keyword1'] == kw1) & (cooccurrence_df['Keyword2'] == kw2)) |
                                    ((cooccurrence_df['Keyword1'] == kw2) & (cooccurrence_df['Keyword2'] == kw1))
                                ]
                                row.append(len(cooc))
                        matrix_data.append(row)
                    
                    if matrix_data:
                        fig_matrix = px.imshow(matrix_data, 
                                             x=top_keywords, y=top_keywords,
                                             title="Matriz de Co-ocorrência (Top 20 Keywords)",
                                             color_continuous_scale='Blues')
                        st.plotly_chart(fig_matrix, use_container_width=True)
                
                # Top pares de co-ocorrência
                st.subheader("🔗 Top Pares de Co-ocorrência")
                top_pairs = cooc_counts.nlargest(15, 'Count')
                top_pairs['Pair'] = top_pairs['Keyword1'] + ' ↔ ' + top_pairs['Keyword2']
                
                fig_pairs = px.bar(top_pairs, x='Count', y='Pair', orientation='h',
                                 title="Top 15 Pares Mais Co-ocorrentes")
                st.plotly_chart(fig_pairs, use_container_width=True)
                
                # Explicação da co-ocorrência
                st.markdown("""
                <div class="explanation-box">
                <h4>🔍 Como Interpretar Co-ocorrência de Keywords:</h4>
                
                <p><strong>🎯 Degree Centrality (Centralidade de Grau):</strong></p>
                <ul>
                    <li><strong>Alta centralidade:</strong> Keywords que aparecem com muitas outras</li>
                    <li><strong>Significado:</strong> Conceitos "hub" que conectam diferentes áreas</li>
                    <li><strong>Estratégia:</strong> Use para integrar diferentes especialidades</li>
                    <li><strong>Exemplo:</strong> "sustainability" pode conectar química, economia e regulamentação</li>
                </ul>
                
                <p><strong>🌉 Betweenness Centrality:</strong></p>
                <ul>
                    <li><strong>Alta betweenness:</strong> Keywords que fazem "ponte" entre clusters</li>
                    <li><strong>Significado:</strong> Conceitos que conectam áreas distintas</li>
                    <li><strong>Oportunidade:</strong> Temas interdisciplinares para inovação</li>
                    <li><strong>Exemplo:</strong> "nanotechnology" pode conectar materiais e aplicações</li>
                </ul>
                
                <p><strong>🔥 Matriz de Co-ocorrência:</strong></p>
                <ul>
                    <li><strong>Cores intensas:</strong> Combinações frequentes de conceitos</li>
                    <li><strong>Padrões de blocos:</strong> Clusters temáticos bem definidos</li>
                    <li><strong>Células isoladas:</strong> Conexões únicas ou raras</li>
                </ul>
                
                <p><strong>🔗 Top Pares Co-ocorrentes:</strong></p>
                <ul>
                    <li><strong>Pares com >10 ocorrências:</strong> Combinações consolidadas</li>
                    <li><strong>Pares crescentes:</strong> Novas associações emergindo</li>
                    <li><strong>Ausências notáveis:</strong> Oportunidades de conexão</li>
                </ul>
                
                <p><strong>💡 Aplicações Práticas:</strong></p>
                <ul>
                    <li><strong>Desenvolvimento de produtos:</strong> Combine conceitos co-ocorrentes</li>
                    <li><strong>Marketing técnico:</strong> Use pares estabelecidos em comunicação</li>
                    <li><strong>Parcerias:</strong> Conecte especialistas de conceitos relacionados</li>
                    <li><strong>Literatura review:</strong> Explore conexões pouco estudadas</li>
                </ul>
                
                <p><strong>🎯 Exemplo para Tintas:</strong> Se "self-healing" co-ocorre com "microcapsules", 
                desenvolva tintas auto-reparáveis usando encapsulamento de agentes reparadores.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Nenhuma co-ocorrência encontrada com frequência ≥ {min_cooc}")
        else:
            st.warning("⚠️ Dados insuficientes para análise de co-ocorrência")
    
    # ===============================
    # 6. REDES DE COLABORAÇÃO
    # ===============================
    elif "6. Redes de Colaboração" in analysis_option:
        st.markdown('<div class="section-header">🤝 6. Análise de Redes de Colaboração</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Redes de Colaboração?</h4>
        <ul>
            <li><strong>Padrões colaborativos:</strong> Revela conexões entre pesquisadores</li>
            <li><strong>Hubs de expertise:</strong> Identifica centros de conhecimento</li>
            <li><strong>Transferência tecnológica:</strong> Acelera disseminação de inovações</li>
            <li><strong>Oportunidades:</strong> Facilita identificação de novos parceiros</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Rede de colaboração entre autores
        collab_network = create_collaboration_network(df)
        
        if collab_network.number_of_nodes() > 0:
            st.write(f"👥 Rede de colaboração: {collab_network.number_of_nodes()} autores, {collab_network.number_of_edges()} colaborações")
            
            # Métricas de rede
            col1, col2, col3 = st.columns(3)
            
            with col1:
                density = nx.density(collab_network)
                st.metric("Densidade da Rede", f"{density:.3f}")
            
            with col2:
                components = nx.number_connected_components(collab_network)
                st.metric("Componentes Conectados", components)
            
            with col3:
                if collab_network.number_of_nodes() > 1:
                    avg_clustering = nx.average_clustering(collab_network)
                    st.metric("Clustering Médio", f"{avg_clustering:.3f}")
            
            # Top autores por centralidade
            centrality_measures = {
                'Degree': nx.degree_centrality(collab_network),
                'Betweenness': nx.betweenness_centrality(collab_network),
                'Closeness': nx.closeness_centrality(collab_network)
            }
            
            measure_choice = st.selectbox("Selecione medida de centralidade:", 
                                        ['Degree', 'Betweenness', 'Closeness'])
            
            selected_centrality = centrality_measures[measure_choice]
            central_authors = pd.DataFrame([
                {'Author': author, 'Centrality': centrality}
                for author, centrality in selected_centrality.items()
            ]).sort_values('Centrality', ascending=False).head(15)
            
            fig_central_authors = px.bar(central_authors, x='Centrality', y='Author', 
                                       orientation='h',
                                       title=f"Top 15 Autores por {measure_choice} Centrality")
            st.plotly_chart(fig_central_authors, use_container_width=True)
            
            # Análise de países (se disponível)
            if 'Affiliations' in df.columns:
                st.subheader("🌍 Colaboração por Países")
                
                # Extrair países das afiliações
                countries = []
                for affiliation in df['Affiliations'].dropna():
                    # Busca padrões simples de países
                    common_countries = ['USA', 'China', 'Germany', 'Japan', 'UK', 'France', 'Italy', 
                                      'Canada', 'Australia', 'South Korea', 'Netherlands', 'Sweden']
                    for country in common_countries:
                        if country.lower() in str(affiliation).lower():
                            countries.append(country)
                            break
                
                if countries:
                    country_counts = pd.Series(countries).value_counts()
                    
                    fig_countries = px.bar(x=country_counts.values, y=country_counts.index,
                                         orientation='h', title="Produção por País")
                    st.plotly_chart(fig_countries, use_container_width=True)
                    
            # Explicação das redes de colaboração
            st.markdown("""
            <div class="explanation-box">
            <h4>🔍 Como Interpretar Redes de Colaboração:</h4>
            
            <p><strong>📊 Métricas da Rede:</strong></p>
            <ul>
                <li><strong>Densidade (0-1):</strong> >0.1 = rede bem conectada, <0.05 = fragmentada</li>
                <li><strong>Componentes conectados:</strong> Menor número = melhor integração</li>
                <li><strong>Clustering médio:</strong> >0.3 = tendência a formar grupos colaborativos</li>
            </ul>
            
            <p><strong>🎯 Degree Centrality (Colaborações Diretas):</strong></p>
            <ul>
                <li><strong>Top autores:</strong> Hubs de colaboração - potenciais mentores</li>
                <li><strong>Estratégia:</strong> Conecte-se com estes para ampliar rede</li>
                <li><strong>Indicador:</strong> Experiência em liderança de projetos</li>
            </ul>
            
            <p><strong>🌉 Betweenness Centrality (Conectores):</strong></p>
            <ul>
                <li><strong>Papel:</strong> Autores que conectam grupos diferentes</li>
                <li><strong>Valor:</strong> Facilitam transferência de conhecimento entre áreas</li>
                <li><strong>Oportunidade:</strong> Ideais para projetos interdisciplinares</li>
            </ul>
            
            <p><strong>🏃 Closeness Centrality (Proximidade):</strong></p>
            <ul>
                <li><strong>Interpretação:</strong> Rapidez para acessar toda a rede</li>
                <li><strong>Vantagem:</strong> Acesso rápido a informações e recursos</li>
                <li><strong>Estratégia:</strong> Bons para disseminação de inovações</li>
            </ul>
            
            <p><strong>🌍 Análise por Países:</strong></p>
            <ul>
                <li><strong>Países líderes:</strong> Onde buscar parcerias internacionais</li>
                <li><strong>Mercados emergentes:</strong> Oportunidades de pioneirismo</li>
                <li><strong>Regulamentações:</strong> Países com normas avançadas em sustentabilidade</li>
            </ul>
            
            <p><strong>💡 Ações Estratégicas:</strong></p>
            <ul>
                <li><strong>Parcerias primárias:</strong> Colabore com autores de alta centralidade</li>
                <li><strong>Acesso a redes:</strong> Use conectores para entrar em novos grupos</li>
                <li><strong>Expansão geográfica:</strong> Foque nos top 3 países por produção</li>
                <li><strong>Nichos:</strong> Explore países com crescimento acelerado</li>
            </ul>
            
            <p><strong>🎯 Para Tintas Sustentáveis:</strong> Conecte-se com grupos que combinam 
            expertise em materiais + sustentabilidade + aplicações industriais.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Dados insuficientes para análise de colaboração")
    
    # ===============================
    # 7. ANÁLISE DE CO-CITAÇÃO
    # ===============================
    elif "7. Análise de Co-citação" in analysis_option:
        st.markdown('<div class="section-header">📚 7. Análise de Co-citação</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Co-citação?</h4>
        <ul>
            <li><strong>Base teórica:</strong> Identifica referências centrais da área</li>
            <li><strong>Estrutura metodológica:</strong> Mapeia fundamentos conceituais</li>
            <li><strong>Referência robusta:</strong> Auxilia construção de framework teórico</li>
            <li><strong>Evolução do conhecimento:</strong> Mostra como ideias se conectam</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Simulação de análise de co-citação (dados reais requerem referências)
        st.info("📝 **Nota:** Análise de co-citação completa requer dados de referências dos artigos. Aqui apresentamos uma simulação baseada nos artigos mais citados.")
        
        # Usar artigos mais citados como proxy para análise
        top_cited = df.nlargest(20, 'Citations')[['Title', 'Authors', 'Year', 'Citations', 'Journal']]
        
        st.subheader("📈 Top 20 Artigos Mais Citados (Base de Co-citação)")
        
        # Visualização dos top papers
        fig_top_papers = px.bar(top_cited.sort_values('Citations'), 
                              x='Citations', y='Title',
                              orientation='h',
                              title="Artigos Mais Citados (Candidatos para Co-citação)")
        # CORRIGIDO: usar update_yaxes em vez de update_yaxis
        fig_top_papers.update_yaxes(title="")
        fig_top_papers.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_papers, use_container_width=True)
        
        # Análise temporal dos papers fundamentais
        st.subheader("⏰ Distribuição Temporal dos Papers Fundamentais")
        
        yearly_top = top_cited['Year'].value_counts().sort_index()
        fig_yearly_top = px.bar(x=yearly_top.index, y=yearly_top.values,
                              title="Distribuição Anual dos Top Papers")
        fig_yearly_top.update_layout(xaxis_title="Ano", yaxis_title="Número de Papers")
        st.plotly_chart(fig_yearly_top, use_container_width=True)
        
        # Journals dos papers mais citados
        st.subheader("📖 Revistas dos Papers Fundamentais")
        
        journal_top = top_cited['Journal'].value_counts()
        fig_journal_top = px.pie(values=journal_top.values, names=journal_top.index,
                                title="Distribuição por Revista (Top Papers)")
        st.plotly_chart(fig_journal_top, use_container_width=True)
        
        # Tabela detalhada
        st.subheader("📋 Detalhes dos Papers Fundamentais")
        
        display_df = top_cited.copy()
        display_df['Title'] = display_df['Title'].str[:60] + "..."
        display_df['Authors'] = display_df['Authors'].str[:40] + "..."
        
        st.dataframe(display_df, use_container_width=True)
        
        # Explicação da co-citação
        st.markdown("""
        <div class="explanation-box">
        <h4>🔍 Como Interpretar Análise de Co-citação:</h4>
        
        <p><strong>📚 Papers Fundamentais (Top Citados):</strong></p>
        <ul>
            <li><strong>Base teórica:</strong> Artigos que formam a fundação conceitual da área</li>
            <li><strong>Metodologias-chave:</strong> Técnicas e abordagens estabelecidas</li>
            <li><strong>Marcos históricos:</strong> Breakthrough papers que mudaram o campo</li>
            <li><strong>Referências obrigatórias:</strong> Devem estar em qualquer revisão de literatura</li>
        </ul>
        
        <p><strong>⏰ Distribuição Temporal:</strong></p>
        <ul>
            <li><strong>Papers antigos (>10 anos):</strong> Fundamentos teóricos estabelecidos</li>
            <li><strong>Papers recentes (<5 anos):</strong> Direções atuais e emergentes</li>
            <li><strong>Gaps temporais:</strong> Períodos de menor atividade fundacional</li>
            <li><strong>Aceleração recente:</strong> Indica área em rápido desenvolvimento</li>
        </ul>
        
        <p><strong>📖 Revistas Centrais:</strong></p>
        <ul>
            <li><strong>Concentração alta:</strong> Poucas revistas dominam os fundamentos</li>
            <li><strong>Diversificação:</strong> Área interdisciplinar com múltiplas fontes</li>
            <li><strong>Revistas especializadas:</strong> Foco técnico específico</li>
            <li><strong>Revistas gerais:</strong> Impacto amplo e visibilidade</li>
        </ul>
        
        <p><strong>💡 Como Usar para Co-citação Completa:</strong></p>
        <ul>
            <li><strong>Identifique clusters:</strong> Papers citados juntos formam escolas de pensamento</li>
            <li><strong>Evolução conceitual:</strong> Como ideias fundamentais se desenvolveram</li>
            <li><strong>Lacunas teóricas:</strong> Áreas com poucos papers fundamentais</li>
            <li><strong>Oportunidades:</strong> Conectar teorias de diferentes clusters</li>
        </ul>
        
        <p><strong>🎯 Estratégias Baseadas nos Resultados:</strong></p>
        <ul>
            <li><strong>Literatura review:</strong> Inclua todos os top 20 papers como base</li>
            <li><strong>Posicionamento:</strong> Compare sua abordagem com os fundamentos</li>
            <li><strong>Inovação:</strong> Combine conceitos de papers pouco co-citados</li>
            <li><strong>Credibilidade:</strong> Demonstre conhecimento dos clássicos</li>
        </ul>
        
        <p><strong>📝 Para Tintas Sustentáveis:</strong></p>
        <ul>
            <li><strong>Se domínio é "coating performance":</strong> Base sólida em funcionalidade</li>
            <li><strong>Se gaps em "sustainability":</strong> Oportunidade para papers fundacionais</li>
            <li><strong>Revistas diversas:</strong> Área interdisciplinar - abordagem integrada necessária</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ===============================
    # 8. TOP PAPERS (CITAÇÕES)
    # ===============================
    elif "8. Top Papers" in analysis_option:
        st.markdown('<div class="section-header">🏆 8. Análise de Citações Diretas (Top Papers)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Top Papers?</h4>
        <ul>
            <li><strong>Maior impacto:</strong> Aponta artigos mais influentes da área</li>
            <li><strong>Relevância:</strong> Útil para justificar importância da pesquisa</li>
            <li><strong>Posicionamento:</strong> Permite posicionar contribuições próprias</li>
            <li><strong>Benchmarking:</strong> Define padrões de qualidade e impacto</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Análise detalhada de citações
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Citações", f"{df['Citations'].sum():,}")
        with col2:
            st.metric("Média de Citações", f"{df['Citations'].mean():.1f}")
        with col3:
            st.metric("Índice H", f"{h_index}")
        
        # Distribuição de citações
        st.subheader("📊 Distribuição de Citações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CORRIGIDO: usar nbins em vez de bins
            fig_hist = px.histogram(df, x='Citations', nbins=30, 
                                  title="Histograma de Citações")
            fig_hist.add_vline(x=df['Citations'].mean(), line_dash="dash", 
                             annotation_text=f"Média: {df['Citations'].mean():.1f}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(df, y='Citations', title="Box Plot de Citações")
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Top papers detalhado
        st.subheader("🥇 Top 20 Papers Mais Citados")
        
        top_papers = df.nlargest(20, 'Citations')[['Title', 'Authors', 'Year', 'Citations', 'Journal']]
        
        # Gráfico de barras
        fig_top = px.bar(top_papers.sort_values('Citations'), x='Citations', y='Title',
                        orientation='h', color='Year',
                        title="Top 20 Papers por Citações")
        # CORRIGIDO: usar update_yaxes
        fig_top.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_top, use_container_width=True)
        
        # Análise de Pareto
        st.subheader("📈 Análise de Pareto (80/20)")
        
        sorted_citations = df['Citations'].sort_values(ascending=False)
        cumsum_citations = sorted_citations.cumsum()
        total_citations = cumsum_citations.iloc[-1]
        
        pareto_df = pd.DataFrame({
            'Paper_Rank': range(1, len(sorted_citations) + 1),
            'Cumulative_Percentage': (cumsum_citations / total_citations) * 100,
            'Citations': sorted_citations.values
        })
        
        # Encontrar ponto de 80%
        pareto_80_idx = pareto_df[pareto_df['Cumulative_Percentage'] >= 80].index[0]
        papers_80 = pareto_80_idx + 1
        
        fig_pareto = px.line(pareto_df, x='Paper_Rank', y='Cumulative_Percentage',
                           title=f"Curva de Pareto - {papers_80} papers (80% das citações)")
        fig_pareto.add_hline(y=80, line_dash="dash", annotation_text="80%")
        fig_pareto.add_vline(x=papers_80, line_dash="dash", 
                           annotation_text=f"{papers_80} papers")
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        st.info(f"📊 **Insight Pareto:** {papers_80} papers ({papers_80/len(df)*100:.1f}% do total) concentram 80% de todas as citações")
        
        # Tabela interativa dos top papers
        st.subheader("📋 Tabela Detalhada dos Top Papers")
        
        display_top = top_papers.copy()
        display_top.index = range(1, len(display_top) + 1)
        st.dataframe(display_top, use_container_width=True)
        
        # Explicação da análise de citações
        st.markdown(f"""
        <div class="explanation-box">
        <h4>🔍 Como Interpretar Análise de Top Papers:</h4>
        
        <p><strong>📊 Métricas de Impacto:</strong></p>
        <ul>
            <li><strong>Total de citações:</strong> Impacto cumulativo da área</li>
            <li><strong>Média de citações:</strong> Benchmark para avaliar qualidade</li>
            <li><strong>Índice H:</strong> Equilibrio entre produtividade e impacto</li>
        </ul>
        
        <p><strong>📈 Distribuição de Citações:</strong></p>
        <ul>
            <li><strong>Histograma assimétrico:</strong> Normal - poucos papers muito citados</li>
            <li><strong>Cauda longa:</strong> Maioria dos papers tem poucas citações</li>
            <li><strong>Outliers:</strong> Papers breakthrough com impacto excepcional</li>
            <li><strong>Box plot:</strong> Mostra quartis e identifica papers atípicos</li>
        </ul>
        
        <p><strong>🏆 Top Papers (Ranking):</strong></p>
        <ul>
            <li><strong>Top 1-5:</strong> Clássicos absolutos - estudar profundamente</li>
            <li><strong>Top 6-15:</strong> Referencias importantes - conhecer bem</li>
            <li><strong>Top 16-50:</strong> Papers relevantes - familiarizar-se</li>
            <li><strong>Tendência por cor/ano:</strong> Papers recentes subindo rapidamente</li>
        </ul>
        
        <p><strong>📈 Princípio de Pareto (80/20):</strong></p>
        <ul>
            <li><strong>Concentração de impacto:</strong> Poucos papers dominam citações</li>
            <li><strong>Significado:</strong> Qualidade > Quantidade na pesquisa</li>
            <li><strong>Estratégia:</strong> Foque em produzir papers de alto impacto</li>
            <li><strong>Benchmark:</strong> Entre no grupo dos 20% mais citados</li>
        </ul>
        
        <p><strong>💡 Análise de Perfil dos Top Papers:</strong></p>
        <ul>
            <li><strong>Anos mais citados:</strong> Identifica períodos de breakthrough</li>
            <li><strong>Revistas dominantes:</strong> Onde publicar para maior impacto</li>
            <li><strong>Tipos de paper:</strong> Reviews vs. artigos originais vs. métodos</li>
            <li><strong>Temas recorrentes:</strong> Assuntos que geram alto impacto</li>
        </ul>
        
        <p><strong>🎯 Estratégias Baseadas nos Resultados:</strong></p>
        <ul>
            <li><strong>Benchmarking:</strong> Compare seu trabalho com os top papers</li>
            <li><strong>Gaps de citação:</strong> Identifique temas pouco explorados nos tops</li>
            <li><strong>Colaboração:</strong> Conecte-se com autores dos papers mais citados</li>
            <li><strong>Posicionamento:</strong> Cite e construa sobre os fundamentos estabelecidos</li>
        </ul>
        
        <p><strong>📝 Para Tintas Sustentáveis - Ações:</strong></p>
        <ul>
            <li><strong>Se top papers são sobre "performance":</strong> Inovação sustentável deve manter qualidade</li>
            <li><strong>Se poucos sobre "sustainability":</strong> Oportunidade de papers de alto impacto</li>
            <li><strong>Reviews bem citadas:</strong> Considere escrever review abrangente</li>
            <li><strong>Métodos novos:</strong> Desenvolva técnicas inovadoras de caracterização</li>
        </ul>
        
        <p><strong>🎯 Meta de Impacto:</strong> Almeje estar entre os top 20% mais citados da área 
        (acima de {df['Citations'].quantile(0.8):.0f} citações com base nos dados atuais).</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===============================
    # 9. LEI DE LOTKA
    # ===============================
    elif "9. Lei de Lotka" in analysis_option:
        st.markdown('<div class="section-header">📐 9. Análise de Autores & Lei de Lotka</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="explanation-box">
        <h4>🎯 Por que usar Lei de Lotka?</h4>
        <ul>
            <li><strong>Pesquisadores-chave:</strong> Identifica autores mais produtivos</li>
            <li><strong>Padrões de produtividade:</strong> Revela distribuição estatística</li>
            <li><strong>Redes de influência:</strong> Mapeia hierarquias acadêmicas</li>
            <li><strong>Coautores potenciais:</strong> Facilita identificação de parceiros</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Processar dados de autores
        all_authors = []
        for authors_str in df['Authors'].dropna():
            authors = [a.strip() for a in str(authors_str).split(';')]
            all_authors.extend(authors)
        
        author_productivity = pd.Series(all_authors).value_counts()
        author_stats_df = pd.DataFrame({
            'Author': author_productivity.index,
            'Papers_Count': author_productivity.values
        })
        
        # Calcular citações por autor
        author_citations = {}
        for _, row in df.iterrows():
            if pd.notna(row['Authors']):
                authors = [a.strip() for a in str(row['Authors']).split(';')]
                for author in authors:
                    if author in author_citations:
                        author_citations[author] += row['Citations']
                    else:
                        author_citations[author] = row['Citations']
        
        author_stats_df['Total_Citations'] = author_stats_df['Author'].map(author_citations).fillna(0)
        author_stats_df['Avg_Citations'] = (author_stats_df['Total_Citations'] / 
                                          author_stats_df['Papers_Count']).round(2)
        
        # Métricas gerais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Autores Únicos", len(author_stats_df))
        with col2:
            st.metric("Autor Mais Produtivo", f"{author_stats_df.iloc[0]['Papers_Count']} papers")
        with col3:
            avg_papers = author_stats_df['Papers_Count'].mean()
            st.metric("Média Papers/Autor", f"{avg_papers:.1f}")
        
        # Top autores
        st.subheader("🏆 Top 20 Autores Mais Produtivos")
        
        top_authors = author_stats_df.head(20)
        
        fig_authors = px.bar(top_authors, x='Papers_Count', y='Author', 
                           orientation='h', color='Total_Citations',
                           title="Produtividade dos Autores")
        # CORRIGIDO: usar update_yaxes
        fig_authors.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_authors, use_container_width=True)
        
        # Análise da Lei de Lotka
        st.subheader("📐 Análise da Lei de Lotka")
        
        productivity_counts, frequencies, alpha = analyze_lotka_law(author_stats_df)
        
        if alpha is not None:
            # Gráfico log-log da Lei de Lotka
            fig_lotka = px.scatter(x=productivity_counts, y=frequencies,
                                 title=f"Lei de Lotka (α = {alpha:.2f})",
                                 log_x=True, log_y=True)
            fig_lotka.update_layout(
                xaxis_title="Número de Papers (log)",
                yaxis_title="Número de Autores (log)"
            )
            
            # Adicionar linha teórica
            x_theory = np.logspace(0, np.log10(max(productivity_counts)), 100)
            y_theory = frequencies[0] * (x_theory[0] / x_theory) ** alpha
            
            fig_lotka.add_scatter(x=x_theory, y=y_theory, mode='lines',
                                name=f'Lei de Lotka (α={alpha:.2f})',
                                line=dict(dash='dash'))
            
            st.plotly_chart(fig_lotka, use_container_width=True)
            
            # Interpretação
            st.markdown(f"""
            <div class="metric-card">
            <h4>🔍 Interpretação da Lei de Lotka</h4>
            <p><strong>Expoente α = {alpha:.2f}</strong></p>
            <ul>
                <li>α ≈ 2: Distribuição clássica de Lotka (poucos autores muito produtivos)</li>
                <li>α > 2: Concentração ainda maior nos top autores</li>
                <li>α < 2: Distribuição mais igualitária</li>
            </ul>
            <p><strong>Seu resultado ({alpha:.2f}):</strong> {'Concentração alta' if alpha > 2 else 'Concentração moderada' if alpha > 1.5 else 'Distribuição equilibrada'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Distribuição de produtividade
        st.subheader("📊 Distribuição de Produtividade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CORRIGIDO: usar nbins
            fig_dist = px.histogram(author_stats_df, x='Papers_Count', nbins=20,
                                  title="Distribuição de Papers por Autor")
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Estatísticas de produtividade
            prod_stats = author_stats_df['Papers_Count'].describe()
            
            st.markdown("""
            **📈 Estatísticas de Produtividade:**
            """)
            for stat, value in prod_stats.items():
                st.write(f"- **{stat.title()}:** {value:.1f}")
        
        # Tabela dos top autores
        st.subheader("📋 Detalhes dos Top Autores")
        
        display_authors = top_authors.copy()
        display_authors.index = range(1, len(display_authors) + 1)
        display_authors['Author'] = display_authors['Author'].str[:40] + "..."
        
        st.dataframe(display_authors, use_container_width=True)
        
        # Explicação da Lei de Lotka
        st.markdown(f"""
        <div class="explanation-box">
        <h4>🔍 Como Interpretar Lei de Lotka & Análise de Autores:</h4>
        
        <p><strong>📐 Lei de Lotka (α = {alpha if alpha else 'N/A':.2f}):</strong></p>
        <ul>
            <li><strong>α ≈ 2.0:</strong> Distribuição clássica - poucos autores muito produtivos</li>
            <li><strong>α > 2.0:</strong> Concentração ainda maior nos top autores</li>
            <li><strong>α < 2.0:</strong> Distribuição mais equilibrada entre autores</li>
            <li><strong>Seu resultado ({alpha if alpha else 'N/A':.2f}):</strong> {'Concentração alta nos top autores' if alpha and alpha > 2 else 'Concentração moderada' if alpha and alpha > 1.5 else 'Distribuição mais equilibrada'}</li>
        </ul>
        
        <p><strong>🏆 Hierarquia de Produtividade:</strong></p>
        <ul>
            <li><strong>Top 1-5 autores:</strong> Elite científica - líderes estabelecidos</li>
            <li><strong>Top 6-20:</strong> Pesquisadores senior - potenciais mentores</li>
            <li><strong>Cauda longa:</strong> Maioria publica poucos papers - oportunidade</li>
        </ul>
        
        <p><strong>📊 Métricas de Autor:</strong></p>
        <ul>
            <li><strong>Papers por autor:</strong> Produtividade bruta</li>
            <li><strong>Citações totais:</strong> Impacto cumulativo</li>
            <li><strong>Citações médias:</strong> Qualidade por paper</li>
            <li><strong>Combinação ideal:</strong> Alta produtividade + Alto impacto</li>
        </ul>
        
        <p><strong>📈 Distribuição de Produtividade:</strong></p>
        <ul>
            <li><strong>Moda baixa:</strong> Maioria dos autores publica pouco</li>
            <li><strong>Outliers:</strong> Autores excepcionalmente produtivos</li>
            <li><strong>Mediana vs Média:</strong> Assimetria indica concentração</li>
        </ul>
        
        <p><strong>💡 Estratégias por Nível de Autor:</strong></p>
        
        <p><strong>🎯 Para Pesquisadores Iniciantes:</strong></p>
        <ul>
            <li><strong>Objetivo:</strong> Entrar no top 50% (>{author_stats_df['Papers_Count'].quantile(0.5):.1f} papers)</li>
            <li><strong>Estratégia:</strong> Colabore com autores produtivos</li>
            <li><strong>Foco:</strong> Qualidade > Quantidade inicialmente</li>
        </ul>
        
        <p><strong>🎯 Para Pesquisadores Intermediários:</strong></p>
        <ul>
            <li><strong>Objetivo:</strong> Top 20 autores da área</li>
            <li><strong>Estratégia:</strong> Lidere projetos colaborativos</li>
            <li><strong>Foco:</strong> Estabeleça programa de pesquisa consistente</li>
        </ul>
        
        <p><strong>🎯 Para Líderes Estabelecidos:</strong></p>
        <ul>
            <li><strong>Objetivo:</strong> Manter posição top 5</li>
            <li><strong>Estratégia:</strong> Mentorar novos pesquisadores</li>
            <li><strong>Foco:</strong> Papers de alto impacto e reviews influentes</li>
        </ul>
        
        <p><strong>🤝 Identificação de Parceiros:</strong></p>
        <ul>
            <li><strong>Alta produtividade + Baixo impacto:</strong> Precisam melhorar qualidade</li>
            <li><strong>Baixa produtividade + Alto impacto:</strong> Foco em qualidade</li>
            <li><strong>Crescimento rápido:</strong> Estrelas em ascensão</li>
            <li><strong>Veteranos estáveis:</strong> Experiência e redes estabelecidas</li>
        </ul>
        
        <p><strong>📝 Para Tintas Sustentáveis - Ações:</strong></p>
        <ul>
            <li><strong>Conecte-se com top 10:</strong> Para projetos de alto impacto</li>
            <li><strong>Monitore emergentes:</strong> Colaborações futuras promissoras</li>
            <li><strong>Diversidade geográfica:</strong> Amplie rede internacional</li>
            <li><strong>Especialização complementar:</strong> Combine expertise química + aplicada</li>
        </ul>
        
        <p><strong>🎯 Meta Pessoal:</strong> Para entrar no top 20% da área, almeje >{author_stats_df['Papers_Count'].quantile(0.8)} papers 
        com impacto médio de >{author_stats_df['Avg_Citations'].quantile(0.8):.1f} citações por paper.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===============================
    # RESUMO EXECUTIVO
    # ===============================
    
    # Seção de Insights Gerais
    st.markdown('<div class="section-header">💡 Resumo Executivo & Insights Estratégicos</div>', unsafe_allow_html=True)
    
    with st.expander("📊 Resumo Executivo - Todas as Análises", expanded=True):
        st.markdown(f"""
        ### 🎯 Principais Descobertas
        
        **📈 Estado da Arte:**
        - **Volume de pesquisa:** {len(df)} artigos analisados
        - **Impacto científico:** {df['Citations'].sum()} citações totais (H-index: {h_index})
        - **Período coberto:** {df['Year'].min()}-{df['Year'].max()}
        - **Colaboração:** {collaboration_rate*100:.1f}% dos papers são colaborativos
        
        **🌱 Evolução Temática:**
        - **Temas emergentes:** Identifique trends positivos para investimento
        - **Áreas maduras:** Aproveite conhecimento consolidado
        - **Lacunas temporais:** Oportunidades para pesquisa pioneira
        
        **🗺️ Posicionamento Estratégico:**
        - **Motor Themes:** Áreas para liderança e diferenciação
        - **Basic Themes:** Fundamentos para especialização
        - **Niche Themes:** Mercados rentáveis específicos
        - **Emerging Themes:** Apostas para o futuro
        
        **🔗 Redes de Conhecimento:**
        - **Conceitos centrais:** Use para integrar especialidades
        - **Bridges:** Explore conexões interdisciplinares
        - **Clusters:** Identifique comunidades de prática
        
        **🤝 Ecossistema de Colaboração:**
        - **Hubs de expertise:** Conecte-se para ampliar capacidades
        - **Conectores de rede:** Acesse diferentes comunidades
        - **Especialistas por país:** Oportunidades globais
        
        ### 🎯 Recomendações Estratégicas
        
        **Para P&D (Pesquisa & Desenvolvimento):**
        1. **Investimento Principal (70%):** Foque em Motor Themes identificados
        2. **Diferenciação (20%):** Explore Basic Themes pouco desenvolvidos
        3. **Inovação Radical (10%):** Aposte em Emerging Themes promissores
        
        **Para Parcerias Acadêmicas:**
        1. **Conecte-se** com autores de alta centralidade
        2. **Colabore** com bridges entre diferentes áreas
        3. **Monitore** pesquisadores emergentes em crescimento rápido
        
        **Para Publicação Científica:**
        1. **Journals alvo:** Foque nas top 5 revistas da área
        2. **Timing:** Entre em temas emergentes antes do pico
        3. **Impacto:** Almeje top 20% em citações (>{df['Citations'].quantile(0.8):.0f} citações)
        
        **Para Inovação Tecnológica:**
        1. **Combine** conceitos co-ocorrentes frequentemente
        2. **Explore** conexões raras entre temas
        3. **Antecipe** tendências com base na evolução temporal
        """)
    
    with st.expander("🔬 Insights Específicos para Tintas e Revestimentos Sustentáveis"):
        st.markdown("""
        ### 🌿 Oportunidades Identificadas
        
        **Materiais Bio-baseados:**
        - **Tendência:** Crescimento consistente em polímeros naturais
        - **Ação:** Desenvolva parcerias com fornecedores de biomassa
        - **Timing:** Mercado em expansão - entre agora
        
        **Economia Circular:**
        - **Gap identificado:** Poucos estudos sobre reciclabilidade de tintas
        - **Oportunidade:** Lidere desenvolvimento de tintas 100% recicláveis
        - **Diferencial:** Combine performance + sustentabilidade
        
        **Funcionalidades Inteligentes:**
        - **Emergência:** Auto-reparação e propriedades adaptativas
        - **Tecnologia:** Microencapsulamento e nanomateriais seguros
        - **Mercado:** Premium pricing para funcionalidades avançadas
        
        **Regulamentação Ambiental:**
        - **Driving force:** Normas cada vez mais restritivas
        - **Oportunidade:** Antecipe-se às regulamentações futuras
        - **Vantagem:** First-mover advantage em compliance
        
        ### 🎯 Roadmap Tecnológico Sugerido
        
        **Curto Prazo (1-2 anos):**
        1. **Formulações water-based** com performance equivalent solvent-based
        2. **Redução de VOCs** para <50g/L em todas as categorias
        3. **Parcerias** com universidades top da área
        
        **Médio Prazo (3-5 anos):**
        1. **Tintas auto-reparáveis** para aplicações específicas
        2. **Conteúdo reciclado** >30% sem perda de qualidade
        3. **Certificações** ambientais reconhecidas internacionalmente
        
        **Longo Prazo (5+ anos):**
        1. **Economia circular completa** - tintas 100% recicláveis
        2. **Smart coatings** com funcionalidades sensoriais
        3. **Liderança global** em sustentabilidade do setor
        
        ### 💡 Indicadores de Sucesso
        
        **Métricas de Inovação:**
        - **Patentes:** >5 por ano em sustentabilidade
        - **Publicações:** Top 20% em impacto científico
        - **Parcerias:** >3 universidades de ponta
        
        **Métricas de Mercado:**
        - **Market share:** Liderança em segmento sustentável
        - **Premium pricing:** 10-15% acima da média
        - **Customer satisfaction:** >95% em performance + sustentabilidade
        
        **Métricas Ambientais:**
        - **Carbon footprint:** Redução 50% até 2030
        - **Waste reduction:** 80% dos materiais reaproveitados
        - **LCA score:** Melhor da categoria em todas as métricas
        """)
        
    # Footer com call-to-action
    st.markdown("""
    ---
    <div style="text-align: center; padding: 2rem; background-color: #f0f9ff; border-radius: 0.5rem; margin: 2rem 0;">
        <h3>🚀 Próximos Passos</h3>
        <p><strong>Use estas análises para:</strong></p>
        <p>✅ Definir estratégia de P&D • ✅ Identificar parceiros • ✅ Priorizar investimentos</p>
        <p>✅ Orientar publicações • ✅ Antecipar tendências • ✅ Liderar inovação sustentável</p>
        <hr>
        <p><em>Dashboard Bibliométrico Avançado - Todas as 9 análises científicas incluídas!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()