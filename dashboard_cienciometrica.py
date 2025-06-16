import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Configurações da página
st.set_page_config(
    page_title="Dashboard Cientométrica - Tintas e Revestimentos Sustentáveis",
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
    .warning-box {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funções de análise (adaptadas do script original)
@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados reais do Scopus"""
    try:
        # Carregar dados reais do Scopus
        df = pd.read_csv('scopus.csv', encoding='utf-8')
        
        # Renomear colunas para padronizar
        column_mapping = {
            'Cited by': 'Citations',
            'Author Keywords': 'Author_Keywords',
            'Index Keywords': 'Index_Keywords',
            'Source title': 'Journal',
            'Document Type': 'Document_Type'
        }
        df = df.rename(columns=column_mapping)
        
        # Limpeza e processamento dos dados
        df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce').fillna(0).astype(int)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
        df['Author_Keywords'] = df['Author_Keywords'].fillna('')
        df['Index_Keywords'] = df['Index_Keywords'].fillna('')
        df['Journal'] = df['Journal'].fillna('Não informado')
        df['Authors'] = df['Authors'].fillna('')
        
        # Remover registros sem ano válido
        df = df[df['Year'] > 1900]
        
        st.success(f"✅ Dados reais carregados: {len(df)} artigos do Scopus")
        
    except FileNotFoundError:
        st.error("❌ Arquivo scopus.csv não encontrado! Usando dados sintéticos para demonstração.")
        # Fallback para dados sintéticos
        df = create_synthetic_data()
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}. Usando dados sintéticos.")
        df = create_synthetic_data()
    
    return df

def create_synthetic_data():
    """Cria dados sintéticos como fallback"""
    np.random.seed(42)
    n_papers = 500

    # Dados sintéticos para demonstração
    years = np.random.choice(range(2015, 2025), n_papers, p=[0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.15, 0.1, 0.05, 0.02])
    citations = np.random.exponential(3, n_papers).astype(int)

    authors_pool = [
        "Silva, J.A.", "Santos, M.B.", "Oliveira, C.D.", "Ferreira, L.M.", "Costa, R.P.",
        "Almeida, A.S.", "Rodrigues, P.H.", "Martins, F.G.", "Pereira, D.L.", "Carvalho, T.N.",
        "Lima, V.O.", "Gomes, H.R.", "Ribeiro, S.C.", "Barbosa, E.M.", "Monteiro, K.F.",
        "Rocha, B.D.", "Dias, N.S.", "Cardoso, W.P.", "Melo, I.T.", "Nascimento, Q.V."
    ]

    journals_pool = [
        "Progress in Organic Coatings", "Journal of Coatings Technology", "Surface and Coatings Technology",
        "Applied Surface Science", "Materials Chemistry and Physics", "Green Chemistry",
        "Journal of Cleaner Production", "Environmental Science & Technology", "Polymer Chemistry",
        "Advanced Materials", "ACS Applied Materials & Interfaces", "Chemical Engineering Journal"
    ]

    keywords_pool = [
        "sustainable coatings", "eco-friendly paint", "green chemistry", "bio-based materials", "recycled content",
        "waterborne coatings", "low VOC", "renewable resources", "biodegradable polymers", "life cycle assessment",
        "environmental impact", "circular economy", "sustainable development", "green technology", "eco-design",
        "natural pigments", "plant-based resins", "recycling", "waste reduction", "carbon footprint",
        "energy efficiency", "sustainable manufacturing", "green solvents", "biopolymers", "environmental chemistry"
    ]

    data = []
    for i in range(n_papers):
        # Gerar autores (1-5 autores por paper)
        n_authors = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        authors = np.random.choice(authors_pool, n_authors, replace=False)
        authors_str = "; ".join(authors)

        # Gerar keywords (2-6 keywords por paper)
        n_keywords = np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.3, 0.4, 0.15, 0.05])
        keywords = np.random.choice(keywords_pool, n_keywords, replace=False)
        keywords_str = "; ".join(keywords)

        data.append({
            'Title': f"Sustainable Coating Research {i+1}: {keywords[0].title()}",
            'Authors': authors_str,
            'Year': years[i],
            'Citations': citations[i],
            'Journal': np.random.choice(journals_pool),
            'Author_Keywords': keywords_str,
            'Index_Keywords': keywords_str,
            'Abstract': f"This paper presents a comprehensive study on {keywords[0]}..."
        })

    df = pd.DataFrame(data)
    
    return df

@st.cache_data
def process_authors_and_keywords(df):
    """Processa dados de autores e palavras-chave do DataFrame real"""
    
    # Processar dados de autores
    authors_list = []
    for idx, row in df.iterrows():
        if pd.notna(row['Authors']) and str(row['Authors']).strip():
            authors = [author.strip() for author in str(row['Authors']).split(';')]
            for author in authors:
                if author:
                    authors_list.append({
                        'Paper_ID': idx,
                        'Author': author,
                        'Year': row['Year'],
                        'Citations': row['Citations'],
                        'Journal': row['Journal'],
                        'Title': row['Title']
                    })

    authors_df = pd.DataFrame(authors_list)
    
    # Estatísticas por autor
    if len(authors_df) > 0:
        author_stats = authors_df.groupby('Author').agg({
            'Paper_ID': 'count',
            'Citations': 'sum',
            'Year': ['min', 'max']
        }).round(2)

        author_stats.columns = ['Papers_Count', 'Total_Citations', 'First_Year', 'Last_Year']
        author_stats['Years_Active'] = author_stats['Last_Year'] - author_stats['First_Year'] + 1
        author_stats['Avg_Citations_per_Paper'] = (author_stats['Total_Citations'] / author_stats['Papers_Count']).round(2)
        author_stats = author_stats.sort_values('Papers_Count', ascending=False)
    else:
        author_stats = pd.DataFrame()

    # Processar palavras-chave
    all_keywords = []
    for idx, row in df.iterrows():
        paper_keywords = []
        
        # Keywords dos autores
        if pd.notna(row['Author_Keywords']) and str(row['Author_Keywords']).strip():
            author_kw = [kw.strip().lower() for kw in str(row['Author_Keywords']).split(';')]
            paper_keywords.extend(author_kw)
        
        # Keywords indexadas
        if pd.notna(row['Index_Keywords']) and str(row['Index_Keywords']).strip():
            index_kw = [kw.strip().lower() for kw in str(row['Index_Keywords']).split(';')]
            paper_keywords.extend(index_kw)

        # Remover duplicatas e palavras vazias
        paper_keywords = list(set([kw for kw in paper_keywords if kw and len(kw) > 2]))
        
        for kw in paper_keywords:
            all_keywords.append({
                'Paper_ID': idx,
                'Keyword': kw,
                'Year': row['Year'],
                'Citations': row['Citations']
            })

    keywords_df = pd.DataFrame(all_keywords)
    
    if len(keywords_df) > 0:
        keyword_stats = keywords_df.groupby('Keyword').agg({
            'Paper_ID': 'count',
            'Citations': 'sum'
        }).round(2)
        keyword_stats.columns = ['Papers_Count', 'Total_Citations']
        keyword_stats['Avg_Citations_per_Paper'] = (keyword_stats['Total_Citations'] / keyword_stats['Papers_Count']).round(2)
        keyword_stats = keyword_stats.sort_values('Papers_Count', ascending=False)
    else:
        keyword_stats = pd.DataFrame()

    return authors_df, author_stats, keywords_df, keyword_stats

def calculate_h_index(citations_sorted):
    """Calcula o índice H"""
    h_index = 0
    for i, citations in enumerate(citations_sorted, 1):
        if citations >= i:
            h_index = i
        else:
            break
    return h_index

def main():
    # Título principal
    st.markdown('<div class="main-header">🔬 Dashboard Cientométrica<br>Tintas e Revestimentos Sustentáveis</div>', unsafe_allow_html=True)
    
    # Carregamento dos dados com informações
    st.markdown("### 📊 Carregamento dos Dados")
    
    with st.spinner("Carregando dados do Scopus..."):
        df = load_and_process_data()
        authors_df, author_stats, keywords_df, keyword_stats = process_authors_and_keywords(df)
    
    # Informações sobre o dataset carregado
    st.markdown(f"""
    <div class="metric-card">
    <h4>📈 Informações do Dataset</h4>
    <ul>
        <li><strong>Total de artigos:</strong> {len(df):,}</li>
        <li><strong>Período coberto:</strong> {df['Year'].min()} - {df['Year'].max()}</li>
        <li><strong>Autores únicos:</strong> {len(author_stats):,}</li>
        <li><strong>Palavras-chave únicas:</strong> {len(keyword_stats):,}</li>
        <li><strong>Revistas únicas:</strong> {df['Journal'].nunique():,}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar com informações do estudo
    st.sidebar.header("📋 Informações do Estudo")
    
    # Status dos dados
    try:
        # Verificar se são dados sintéticos (checando se o título contém padrão sintético)
        sample_title = str(df.iloc[0]['Title'])
        is_synthetic = 'Sustainable Coating Research' in sample_title and 'Research Paper' in sample_title
        
        if is_synthetic:
            st.sidebar.error("⚠️ Usando dados sintéticos")
            st.sidebar.info("Coloque o arquivo 'scopus.csv' na mesma pasta para usar dados reais")
        else:
            st.sidebar.success("✅ Usando dados reais do Scopus")
            st.sidebar.info(f"Dataset: {len(df)} artigos carregados")
    except:
        st.sidebar.warning("⚠️ Status dos dados indeterminado")
    
    st.sidebar.markdown("""
    **Base de Dados:** Scopus
    
    **Período:** 2015-2025
    
    **Busca realizada:** 04-06 Abril 2025
    
    **Estratégia de Busca:**
    - ("paint" OR "coat") AND 
    - ("sustainable" OR "eco-friendly" OR "green" OR "environmentally friendly" OR "recycl*")
    
    **Filtros:**
    - Artigos originais
    - Idioma: Inglês
    - Áreas: Ciência dos Materiais, Química, Engenharia
    """)
    
    # Métricas principais
    st.markdown('<div class="section-header">📊 Métricas Gerais</div>', unsafe_allow_html=True)
    
    h_index = calculate_h_index(df['Citations'].sort_values(ascending=False))
    collaboration_rate = (authors_df.groupby('Paper_ID').size() > 1).mean() if len(authors_df) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Artigos", f"{len(df):,}")
    with col2:
        st.metric("Total de Citações", f"{df['Citations'].sum():,}")
    with col3:
        st.metric("Índice H", f"{h_index}")
    with col4:
        st.metric("Taxa de Colaboração", f"{collaboration_rate*100:.1f}%")
    
    # Explicação das métricas
    st.markdown("""
    <div class="explanation-box">
    <h4>🔍 Por que essas métricas são importantes?</h4>
    <ul>
        <li><strong>Total de Artigos:</strong> Indica o volume de pesquisa na área, demonstrando o interesse científico</li>
        <li><strong>Total de Citações:</strong> Reflete o impacto e relevância das pesquisas para a comunidade científica</li>
        <li><strong>Índice H:</strong> Mede simultaneamente produtividade e impacto científico (h artigos com pelo menos h citações cada)</li>
        <li><strong>Taxa de Colaboração:</strong> Mostra o grau de cooperação entre pesquisadores, indicando interdisciplinaridade</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise Temporal
    st.markdown('<div class="section-header">📈 Análise Temporal</div>', unsafe_allow_html=True)
    
    yearly_data = df.groupby('Year').agg({
        'Title': 'count',
        'Citations': ['sum', 'mean']
    }).round(2)
    yearly_data.columns = ['Artigos', 'Citações_Total', 'Citações_Média']
    yearly_data = yearly_data.reset_index()
    
    tab1, tab2 = st.tabs(["Produção Anual", "Citações por Ano"])
    
    with tab1:
        fig_prod = px.bar(yearly_data, x='Year', y='Artigos', 
                         title='Produção Científica por Ano',
                         color='Artigos', color_continuous_scale='viridis')
        fig_prod.update_layout(height=400)
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with tab2:
        fig_cit = px.line(yearly_data, x='Year', y='Citações_Total', 
                         title='Citações Totais por Ano', markers=True)
        fig_cit.update_layout(height=400)
        st.plotly_chart(fig_cit, use_container_width=True)
    
    st.markdown("""
    <div class="explanation-box">
    <h4>📊 Interpretação da Análise Temporal</h4>
    <p><strong>Importância:</strong> A análise temporal revela tendências de crescimento na pesquisa sobre tintas sustentáveis, 
    permitindo identificar períodos de maior interesse científico e possíveis correlações com eventos ambientais ou regulamentações.</p>
    
    <p><strong>Como interpretar:</strong></p>
    <ul>
        <li>Crescimento constante indica área em desenvolvimento</li>
        <li>Picos podem estar relacionados a marcos regulatórios ou tecnológicos</li>
        <li>Citações crescentes sugerem maior impacto e maturidade da área</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de Autores
    st.markdown('<div class="section-header">👥 Análise de Autores</div>', unsafe_allow_html=True)
    
    if len(author_stats) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_authors = author_stats.head(15)
            fig_authors = px.bar(top_authors, y=top_authors.index, x='Papers_Count', 
                                orientation='h', title='Top 15 Autores Mais Produtivos',
                                color='Total_Citations', color_continuous_scale='plasma')
            fig_authors.update_layout(height=500)
            st.plotly_chart(fig_authors, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Autores**")
            for i, (author, data) in enumerate(author_stats.head(10).iterrows(), 1):
                st.markdown(f"""
                **{i}. {author[:25]}...**
                - Artigos: {int(data['Papers_Count'])}
                - Citações: {int(data['Total_Citations'])}
                - Média: {data['Avg_Citations_per_Paper']:.1f}
                """)
    else:
        st.warning("⚠️ Dados de autores não disponíveis ou insuficientes para análise.")
    
    st.markdown("""
    <div class="explanation-box">
    <h4>🎯 Análise de Produtividade dos Autores</h4>
    <p><strong>Por que é relevante:</strong> Identifica os pesquisadores mais influentes na área, 
    facilitando colaborações e identificando especialistas de referência.</p>
    
    <p><strong>Métricas importantes:</strong></p>
    <ul>
        <li><strong>Número de artigos:</strong> Indica produtividade e dedicação à área</li>
        <li><strong>Total de citações:</strong> Reflete o impacto do trabalho do autor</li>
        <li><strong>Citações por artigo:</strong> Mostra a qualidade média das publicações</li>
        <li><strong>Anos ativos:</strong> Demonstra consistência e experiência na área</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de Citações
    st.markdown('<div class="section-header">📈 Análise de Citações</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(df, x='Citations', nbins=30, 
                               title='Distribuição de Citações')
        fig_hist.add_vline(x=df['Citations'].mean(), line_dash="dash", 
                          annotation_text=f"Média: {df['Citations'].mean():.1f}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        top_cited = df.nlargest(10, 'Citations')
        fig_top = px.bar(top_cited, y='Citations', x=range(1, 11), 
                        title='Top 10 Artigos Mais Citados',
                        hover_data=['Title'])
        fig_top.update_xaxes(title="Ranking")
        st.plotly_chart(fig_top, use_container_width=True)
    
    # Curva de Pareto
    st.subheader("Princípio de Pareto - Concentração de Citações")
    sorted_citations = np.sort(df['Citations'].values)[::-1]
    cumsum_citations = np.cumsum(sorted_citations)
    pareto_data = pd.DataFrame({
        'Artigo': range(1, len(sorted_citations) + 1),
        'Porcentagem_Acumulada': cumsum_citations / cumsum_citations[-1] * 100
    })
    
    fig_pareto = px.line(pareto_data, x='Artigo', y='Porcentagem_Acumulada',
                        title='Curva de Pareto - Concentração de Citações')
    fig_pareto.add_hline(y=80, line_dash="dash", annotation_text="80% das citações")
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    st.markdown("""
    <div class="explanation-box">
    <h4>📊 Análise da Distribuição de Citações</h4>
    <p><strong>Importância:</strong> Revela como o impacto científico está distribuído, 
    identificando artigos de alta influência e padrões de citação típicos da área.</p>
    
    <p><strong>Interpretações:</strong></p>
    <ul>
        <li><strong>Distribuição assimétrica:</strong> Típica em cienciometria, poucos artigos concentram muitas citações</li>
        <li><strong>Princípio de Pareto:</strong> Geralmente 20% dos artigos recebem 80% das citações</li>
        <li><strong>Artigos altamente citados:</strong> Representam marcos ou revisões importantes na área</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de Palavras-chave
    st.markdown('<div class="section-header">🔍 Análise de Palavras-chave</div>', unsafe_allow_html=True)
    
    if len(keyword_stats) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_keywords = keyword_stats.head(20)
            fig_kw = px.bar(top_keywords, y=top_keywords.index, x='Papers_Count',
                           orientation='h', title='Top 20 Palavras-chave Mais Frequentes',
                           color='Total_Citations', color_continuous_scale='viridis')
            fig_kw.update_layout(height=600)
            st.plotly_chart(fig_kw, use_container_width=True)
        
        with col2:
            st.markdown("**Palavras-chave Emergentes**")
            for i, (keyword, data) in enumerate(keyword_stats.head(15).iterrows(), 1):
                st.markdown(f"""
                **{i}. {keyword}**
                - Frequência: {int(data['Papers_Count'])}
                - Citações: {int(data['Total_Citations'])}
                """)
        
        # Nuvem de palavras
        st.subheader("Nuvem de Palavras-chave")
        try:
            word_freq = dict(zip(keyword_stats.index[:30], keyword_stats['Papers_Count'][:30]))
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                max_words=30, colormap='viridis').generate_from_frequencies(word_freq)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.info("💡 Nuvem de palavras não disponível. Instale a biblioteca wordcloud para visualização.")
    else:
        st.warning("⚠️ Dados de palavras-chave não disponíveis ou insuficientes para análise.")
    
    st.markdown("""
    <div class="explanation-box">
    <h4>🏷️ Análise de Palavras-chave</h4>
    <p><strong>Relevância:</strong> Identifica os temas centrais e emergentes na pesquisa sobre tintas sustentáveis, 
    revelando focos de interesse e lacunas de pesquisa.</p>
    
    <p><strong>Insights importantes:</strong></p>
    <ul>
        <li><strong>Frequência alta:</strong> Temas consolidados e de interesse contínuo</li>
        <li><strong>Crescimento temporal:</strong> Identifica tendências emergentes</li>
        <li><strong>Correlações:</strong> Revela campos interdisciplinares</li>
        <li><strong>Impacto por palavra:</strong> Mostra quais temas geram mais citações</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de Revistas
    st.markdown('<div class="section-header">📚 Análise de Revistas</div>', unsafe_allow_html=True)
    
    journal_stats = df.groupby('Journal').agg({
        'Title': 'count',
        'Citations': ['sum', 'mean']
    }).round(2)
    journal_stats.columns = ['Articles', 'Total_Citations', 'Avg_Citations']
    journal_stats = journal_stats.sort_values('Articles', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_journals = journal_stats.head(10)
        fig_journals = px.bar(top_journals, y=top_journals.index, x='Articles',
                             orientation='h', title='Top 10 Revistas por Número de Artigos',
                             color='Avg_Citations', color_continuous_scale='blues')
        fig_journals.update_layout(height=400)
        st.plotly_chart(fig_journals, use_container_width=True)
    
    with col2:
        fig_scatter = px.scatter(journal_stats.head(15), x='Articles', y='Avg_Citations',
                               size='Total_Citations', hover_name=journal_stats.head(15).index,
                               title='Produtividade vs Impacto das Revistas')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Lei de Bradford
    st.subheader("Lei de Bradford - Concentração de Publicações")
    sorted_journals = journal_stats.sort_values('Articles', ascending=False)
    cumsum_articles = sorted_journals['Articles'].cumsum()
    bradford_data = pd.DataFrame({
        'Revista': range(1, len(sorted_journals) + 1),
        'Porcentagem_Acumulada': cumsum_articles / cumsum_articles.iloc[-1] * 100
    })
    
    fig_bradford = px.line(bradford_data, x='Revista', y='Porcentagem_Acumulada',
                          title='Lei de Bradford - Distribuição de Artigos por Revista')
    fig_bradford.add_hline(y=33.33, line_dash="dash", annotation_text="33.3% (Núcleo)")
    fig_bradford.add_hline(y=66.67, line_dash="dash", annotation_text="66.7% (Zona 2)")
    st.plotly_chart(fig_bradford, use_container_width=True)
    
    st.markdown("""
    <div class="explanation-box">
    <h4>📖 Análise de Revistas Científicas</h4>
    <p><strong>Utilidade:</strong> Identifica os principais veículos de publicação na área, 
    orientando pesquisadores sobre onde publicar e quais revistas acompanhar.</p>
    
    <p><strong>Métricas de avaliação:</strong></p>
    <ul>
        <li><strong>Número de artigos:</strong> Indica revistas especializadas na área</li>
        <li><strong>Citações médias:</strong> Reflete o prestígio e impacto da revista</li>
        <li><strong>Lei de Bradford:</strong> Mostra que poucas revistas concentram muitas publicações</li>
        <li><strong>Distribuição geográfica:</strong> Revela centros de pesquisa importantes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Insights e Conclusões
    st.markdown('<div class="section-header">💡 Insights e Recomendações</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>🔬 Tendências Identificadas</h4>
        <ul>
            <li>Crescimento constante na pesquisa sobre sustentabilidade</li>
            <li>Foco em materiais bio-baseados e reciclados</li>
            <li>Interesse crescente em avaliação de ciclo de vida</li>
            <li>Colaboração internacional em aumento</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Oportunidades de Pesquisa</h4>
        <ul>
            <li>Desenvolvimento de novos pigmentos naturais</li>
            <li>Otimização de processos de reciclagem</li>
            <li>Estudos de durabilidade e performance</li>
            <li>Análises econômicas de viabilidade</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Limitações e Considerações</h4>
    <ul>
        <li>Dados limitados ao Scopus - outras bases podem complementar a análise</li>
        <li>Período de 10 anos pode não capturar todas as tendências históricas</li>
        <li>Análise em inglês pode excluir pesquisas regionais importantes</li>
        <li>Métricas de citação podem favorecer artigos mais antigos</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Metodologia detalhada
    with st.expander("🔍 Metodologia Detalhada"):
        st.markdown("""
        ### Estratégia de Busca
        
        **Base de Dados:** Scopus foi escolhido por sua ampla cobertura em ciências aplicadas e engenharia.
        
        **Período:** 2015-2025 captura a década de maior interesse em sustentabilidade.
        
        **Termos de Busca:**
        ```
        TITLE-ABS-KEY(("paint" OR "coating*") AND 
                      ("sustainable" OR "eco-friendly" OR "green" OR 
                       "environmentally friendly" OR "recycl*"))
        ```
        
        **Filtros Aplicados:**
        - Tipo de documento: Artigos originais
        - Idioma: Inglês
        - Áreas temáticas: Materials Science, Chemistry, Engineering, Environmental Science
        - Status: Final publications
        
        ### Métricas Calculadas
        
        1. **Índice H:** h artigos com pelo menos h citações cada
        2. **Taxa de Colaboração:** % de artigos com múltiplos autores
        3. **Distribuição de Bradford:** Concentração de publicações por revista
        4. **Análise de Pareto:** Concentração de citações
        
        ### Limitações
        
        - Viés de idioma (inglês)
        - Cobertura temporal limitada
        - Possível sub-representação de pesquisas regionais
        """)

if __name__ == "__main__":
    main()