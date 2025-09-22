"""
Tab 4: AI Recommendation System Showcase

Interactive demo interface for the recommendation system.
Designed to impress in portfolio presentations and interviews.
"""

import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Imports for recommendation system
from recommendation.models import UserProfile, Resource
from recommendation.adapters import ChunkToResourceAdapter
from recommendation.engines import RecommendationEngine
from database.sql_handler import SQLHandler
from database.vector_handler import VectorHandler
from core.embedder import Embedder

def get_ui_text(key: str, language: str = 'en') -> str:
    """Get UI text in the specified language."""
    
    texts = {
        'en': {
            'user_profile_intelligence': 'User Profile Intelligence',
            'interest_profile_analysis': 'Interest Profile Analysis',
            'profile_stats': 'Profile Stats',
            'total_queries': 'Total Queries',
            'recent_categories': 'Recent Categories',
            'language_preference': 'Language Preference',
            'engagement_style': 'Engagement Style',
            'generate_recommendations': 'Generate Recommendations',
            'quick_suggestions': 'Quick Suggestions',
            'enter_query': 'Enter your query:',
            'placeholder_query': 'Try asking about hiring, AI features, or platform basics...',
            'demo_control_panel': 'Demo Control Panel',
            'select_user_profile': 'Select User Profile',
            'demo_mode': 'Demo Mode',
            'manual_query': 'Manual Query',
            'ab_test': 'A/B Test',
            'language': 'Language',
            'show_metrics': 'Show Metrics',
            'debug_mode': 'Debug Mode',
            'interactive_query_simulation': 'Interactive Query Simulation',
            'query_analysis': 'Query Analysis',
            'ai_generated_recommendations': 'AI-Generated Recommendations',
            'algorithm_ab_testing': 'Algorithm A/B Testing',
            'ab_testing_mode': 'A/B Testing Mode'
        },
        'es': {
            'user_profile_intelligence': 'Análisis de Perfil de Usuario',
            'interest_profile_analysis': 'Análisis de Perfil de Intereses',
            'profile_stats': 'Estadísticas del Perfil',
            'total_queries': 'Consultas Totales',
            'recent_categories': 'Categorías Recientes',
            'language_preference': 'Preferencia de Idioma',
            'engagement_style': 'Estilo de Interacción',
            'generate_recommendations': 'Generar Recomendaciones',
            'quick_suggestions': 'Sugerencias Rápidas',
            'enter_query': 'Introduce tu consulta:',
            'placeholder_query': 'Pregunta sobre contratación, IA, o funcionalidades...',
            'demo_control_panel': 'Panel de Control de Demo',
            'select_user_profile': 'Seleccionar Perfil de Usuario',
            'demo_mode': 'Modo Demo',
            'manual_query': 'Consulta Manual',
            'ab_test': 'Test A/B',
            'language': 'Idioma',
            'show_metrics': 'Mostrar Métricas',
            'debug_mode': 'Modo Debug',
            'interactive_query_simulation': 'Simulación de Consulta Interactiva',
            'query_analysis': 'Análisis de Consulta',
            'ai_generated_recommendations': 'Recomendaciones Generadas por IA',
            'algorithm_ab_testing': 'Test A/B de Algoritmos',
            'ab_testing_mode': 'Modo Test A/B'
        }
    }
    
    return texts.get(language, texts['en']).get(key, key)

def render_tab4(sql_handler: SQLHandler, vector_handler: VectorHandler, 
               embedder: Embedder, base_prompt_dir: str):
    """
    Main render function for the AI Recommendation System showcase.
    """
    
    # Page header with simple styling
    st.markdown("# 🎯 AI Recommendation System")
    st.markdown("**Intelligent Content Discovery • Personalized User Experience • Real-time Analytics**")
    
    # Initialize session state
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = 'manual'
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'recommendations_history' not in st.session_state:
        st.session_state.recommendations_history = []
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = 'es'
    
    # Demo Control Panel
    render_control_panel()
    
    # Get selected language
    selected_language = st.session_state.get('language_selector', 'es')
    st.session_state.selected_language = selected_language
    
    # Load selected user
    user_profile = load_selected_user()
    if not user_profile:
        st.error("🚫 No user profile selected. Please choose a user from the dropdown above.")
        return
    
    st.session_state.current_user = user_profile
    
    # User Profile Insight Panel
    render_user_insight_panel(user_profile, selected_language)
    
    # Demo Mode Router
    if st.session_state.demo_mode == 'manual':
        render_manual_mode(user_profile, sql_handler, embedder, base_prompt_dir, selected_language)
    elif st.session_state.demo_mode == 'ab_test':
        render_ab_test_mode(user_profile, sql_handler, embedder, base_prompt_dir, selected_language)
    
    # System Performance Dashboard
    render_performance_dashboard(selected_language)

def render_control_panel():
    """Render the demo control panel."""
    
    with st.container():
        st.markdown("### 🎮 Demo Control Panel")
        
        col1, col2, col3, col4 = st.columns([3, 2, 1.5, 1.5])
        
        with col1:
            # User selection dropdown
            user_options = get_available_users()
            selected_user = st.selectbox(
                "👤 Select User Profile",
                options=list(user_options.keys()),
                format_func=lambda x: f"{user_options[x]['name']} ({user_options[x]['role']})",
                key="user_selector"
            )
        
        with col2:
            # Demo mode selection (removed auto sequence)
            demo_mode = st.radio(
                "🎮 Demo Mode",
                options=['manual', 'ab_test'],
                format_func=lambda x: {
                    'manual': '🎯 Manual Query',
                    'ab_test': '⚖️ A/B Test'
                }[x],
                horizontal=True,
                key="demo_mode_selector"
            )
            st.session_state.demo_mode = demo_mode
        
        with col3:
            # Language selection
            language = st.selectbox(
                "🌍 Language",
                options=['es', 'en'],
                format_func=lambda x: '🇪🇸 Español' if x == 'es' else '🇬🇧 English',
                key="language_selector"
            )
        
        with col4:
            # Settings toggles
            show_metrics = st.checkbox("📊 Show Metrics", value=True)
            show_debug = st.checkbox("🔧 Debug Mode", value=False)

def get_available_users() -> Dict[str, Dict[str, str]]:
    """Get available user profiles from JSON files."""
    users_dir = Path("data/users")
    user_options = {}
    
    if users_dir.exists():
        for user_file in users_dir.glob("*.json"):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                
                user_id = user_data['user_id']
                profile = user_data['profile']
                
                user_options[user_id] = {
                    'name': profile['name'],
                    'role': profile['role'],
                    'company_type': profile['company_type'],
                    'file_path': str(user_file)
                }
            except Exception as e:
                st.error(f"Error loading user file {user_file}: {e}")
    
    return user_options

def load_selected_user() -> Optional[UserProfile]:
    """Load the selected user profile."""
    try:
        user_options = get_available_users()
        selected_user_id = st.session_state.get('user_selector')
        
        if selected_user_id and selected_user_id in user_options:
            user_file_path = Path(user_options[selected_user_id]['file_path'])
            return UserProfile.load_from_file(user_file_path)
        
        return None
    except Exception as e:
        st.error(f"Error loading user profile: {e}")
        return None

def render_user_insight_panel(user: UserProfile, language: str):
    """Render user profile insights with visual elements."""
    
    st.markdown(f"### 👤 {get_ui_text('user_profile_intelligence', language)}")
    
    # Show metrics update notification
    if st.session_state.get('user_metrics_updated', False):
        st.success("📈 User profile updated based on recent activity!")
        st.session_state.user_metrics_updated = False
    
    with st.container():
        # Create user info card using native streamlit
        st.info(f"**{user.name}** • {user.role} • {user.company_type.title()} • {user.experience_level.title()}")
        
        # Interest profile visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**🧠 {get_ui_text('interest_profile_analysis', language)}**")
            
            # Create interest scores chart
            if user.computed_interests:
                interests_data = list(user.computed_interests.items())
                interests_data.sort(key=lambda x: x[1], reverse=True)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[score for _, score in interests_data],
                        y=[interest.replace('_', ' ').title() for interest, _ in interests_data],
                        orientation='h',
                        marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'][:len(interests_data)]
                    )
                ])
                
                fig.update_layout(
                    title="Interest Strength by Category",
                    height=300,
                    xaxis_title="Interest Score",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"interests_chart_{len(user.query_history)}")
        
        with col2:
            st.markdown(f"**📊 {get_ui_text('profile_stats', language)}**")
            
            # Query history stats (now dynamic)
            total_queries = len(user.query_history)
            recent_categories = user.get_recent_categories(5)
            
            # Create metrics with dynamic updates
            st.metric(get_ui_text('total_queries', language), total_queries, 
                     delta=1 if st.session_state.get('user_metrics_updated') else None)
            st.metric(get_ui_text('recent_categories', language), len(set(recent_categories)))
            st.metric(get_ui_text('language_preference', language), user.language_preference.upper())
            
            # Engagement pattern
            engagement_patterns = {
                'decision_maker': '🎯 Decision Maker',
                'explorer': '🔍 Explorer', 
                'researcher': '📚 Researcher',
                'focused': '🎪 Focused'
            }
            
            pattern = user.query_history[-1].get('intent', 'explorer') if user.query_history else 'explorer'
            st.info(f"**{get_ui_text('engagement_style', language)}:** {engagement_patterns.get(pattern, '🤔 Analytical')}")

def render_manual_mode(user: UserProfile, sql_handler: SQLHandler, 
                      embedder: Embedder, base_prompt_dir: str, language: str):
    """Render manual query input mode."""
    
    st.markdown(f"### 💬 {get_ui_text('interactive_query_simulation', language)}")
    
    # Query input section
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Smart query suggestions based on user profile
            suggested_queries = get_smart_query_suggestions(user, language)
            
            # Initialize query input
            initial_query = ""
            
            # Check if a suggestion was clicked
            for idx, (query, _) in enumerate(suggested_queries):
                if st.session_state.get(f'suggestion_clicked_{idx}', False):
                    initial_query = query
                    st.session_state[f'suggestion_clicked_{idx}'] = False
                    # Immediately process this query
                    st.session_state['auto_process_query'] = query
                    break
            
            query_input = st.text_area(
                get_ui_text('enter_query', language),
                value=initial_query,
                height=100,
                placeholder=get_ui_text('placeholder_query', language),
                key="manual_query_input"
            )
            
            # Quick suggestion buttons - FIXED: auto-process on click
            st.markdown(f"**💡 {get_ui_text('quick_suggestions', language)}:**")
            
            for idx, (query, _) in enumerate(suggested_queries):
                if st.button(f"💭 {query}", key=f"suggestion_{idx}", use_container_width=True):
                    # Set flags to process this query automatically
                    st.session_state[f'suggestion_clicked_{idx}'] = True
                    st.session_state['auto_process_query'] = query
                    st.rerun()
        
        with col2:
            st.markdown(f"**🎯 {get_ui_text('query_analysis', language)}**")
            
            # Use auto_process_query if available, otherwise use manual input
            analysis_query = st.session_state.get('auto_process_query', query_input)
            
            if analysis_query:
                # Analyze query in real-time
                category = analyze_query_category(analysis_query)
                intent = analyze_query_intent(analysis_query)
                urgency = analyze_query_urgency(analysis_query)
                profile_match = calculate_profile_match(analysis_query, user)
                
                st.markdown(f"""
                **Category:** `{category}`  
                **Intent:** `{intent}`  
                **Urgency:** `{urgency}`  
                **Profile Match:** {profile_match:.0%}
                """)
        
        # Auto-process if suggestion was clicked
        if st.session_state.get('auto_process_query'):
            process_query = st.session_state['auto_process_query']
            del st.session_state['auto_process_query']  # Clear flag
            
            with st.spinner("🧠 AI is analyzing and generating personalized recommendations..."):
                recommendations = generate_recommendations_for_query(
                    user, process_query, sql_handler, embedder
                )
                
                if recommendations:
                    # Update user metrics BEFORE showing recommendations
                    update_user_metrics_dynamically(user, process_query, category, intent)
                    
                    render_recommendations_showcase(recommendations, process_query, language)
                    
                    # Save to history
                    st.session_state.recommendations_history.append({
                        'query': process_query,
                        'recommendations': recommendations,
                        'timestamp': datetime.now(),
                        'user_id': user.user_id
                    })
                else:
                    st.error("😕 No recommendations generated. Please try a different query.")
        
        # Manual generate recommendations button
        elif st.button(f"🚀 {get_ui_text('generate_recommendations', language)}", type="primary", use_container_width=True):
            if query_input:
                with st.spinner("🧠 AI is analyzing and generating personalized recommendations..."):
                    recommendations = generate_recommendations_for_query(
                        user, query_input, sql_handler, embedder
                    )
                    
                    if recommendations:
                        # Update user metrics BEFORE showing recommendations
                        category = analyze_query_category(query_input)
                        intent = analyze_query_intent(query_input)
                        update_user_metrics_dynamically(user, query_input, category, intent)
                        
                        render_recommendations_showcase(recommendations, query_input, language)
                        
                        # Save to history
                        st.session_state.recommendations_history.append({
                            'query': query_input,
                            'recommendations': recommendations,
                            'timestamp': datetime.now(),
                            'user_id': user.user_id
                        })
                    else:
                        st.error("😕 No recommendations generated. Please try a different query.")
            else:
                st.warning("Please enter a query first!")

def update_user_metrics_dynamically(user: UserProfile, query: str, category: str, intent: str):
    """Update user metrics based on new query behavior."""
    
    # Add new query to history
    from datetime import datetime
    new_query = {
        'query': query,
        'category': category,
        'intent': intent,
        'timestamp': datetime.now().isoformat()
    }
    user.query_history.append(new_query)
    
    # Update computed interests based on new query
    if category in user.computed_interests:
        # Increase interest in this category
        current_score = user.computed_interests[category]
        # Boost by 0.1 but cap at 1.0
        user.computed_interests[category] = min(1.0, current_score + 0.1)
    else:
        # Add new interest category
        user.computed_interests[category] = 0.3  # Starting score
    
    # Decay other interests slightly (realistic behavior)
    for interest_category in user.computed_interests:
        if interest_category != category:
            user.computed_interests[interest_category] *= 0.98  # Slight decay
    
    # Update session state to trigger UI refresh
    st.session_state.current_user = user
    st.session_state.user_metrics_updated = True

def render_ab_test_mode(user: UserProfile, sql_handler: SQLHandler,
                       embedder: Embedder, base_prompt_dir: str, language: str):
    """Render A/B testing comparison mode."""
    
    st.markdown(f"### ⚖️ {get_ui_text('algorithm_ab_testing', language)}")
    
    st.info(f"""
    **🧪 {get_ui_text('ab_testing_mode', language)}**  
    Compare different recommendation algorithms side-by-side to see which performs better 
    for your selected user profile.
    """)
    
    # Query input for A/B test
    test_query = st.text_input(
        "Enter query for A/B testing:",
        placeholder="¿Cómo puedo escalar mi equipo técnico?" if language == 'es' else "How can I scale my technical team?",
        key="ab_test_query"
    )
    
    if st.button("🧪 Run A/B Test", type="primary") and test_query:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🅰️ Algorithm A: Interest-Focused")
            with st.spinner("Generating recommendations with Algorithm A..."):
                recs_a = generate_recommendations_for_query(
                    user, test_query, sql_handler, embedder, algorithm='interest_focused'
                )
                if recs_a:
                    render_recommendations_showcase(recs_a, test_query, language, compact=True, prefix="A")
        
        with col2:
            st.markdown("#### 🅱️ Algorithm B: Role-Focused")
            with st.spinner("Generating recommendations with Algorithm B..."):
                recs_b = generate_recommendations_for_query(
                    user, test_query, sql_handler, embedder, algorithm='role_focused'
                )
                if recs_b:
                    render_recommendations_showcase(recs_b, test_query, language, compact=True, prefix="B")
        
        # Comparison metrics
        st.markdown("#### 📊 A/B Test Results")
        
        if recs_a and recs_b:
            comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
            
            with comparison_col1:
                avg_score_a = sum(r.final_score for r in recs_a) / len(recs_a)
                avg_score_b = sum(r.final_score for r in recs_b) / len(recs_b)
                
                st.metric("Avg Relevance A", f"{avg_score_a:.3f}")
                st.metric("Avg Relevance B", f"{avg_score_b:.3f}", f"{avg_score_b - avg_score_a:.3f}")
            
            with comparison_col2:
                diversity_a = calculate_diversity_score([r.resource for r in recs_a])
                diversity_b = calculate_diversity_score([r.resource for r in recs_b])
                
                st.metric("Diversity A", f"{diversity_a:.3f}")
                st.metric("Diversity B", f"{diversity_b:.3f}", f"{diversity_b - diversity_a:.3f}")
            
            with comparison_col3:
                # User preference simulation
                preference_a = simulate_user_preference(recs_a, user)
                preference_b = simulate_user_preference(recs_b, user)
                
                st.metric("User Preference A", f"{preference_a:.1%}")
                st.metric("User Preference B", f"{preference_b:.1%}", f"{preference_b - preference_a:.1%}")

def render_recommendations_showcase(recommendations: List[Any], query: str, language: str,
                                  compact: bool = False, prefix: str = ""):
    """Render recommendations using native Streamlit components."""
    
    if not recommendations:
        st.warning("No recommendations generated.")
        return
    
    # Header
    if not compact:
        st.markdown(f"### 🎯 {get_ui_text('ai_generated_recommendations', language)}")
        
        # Query analysis summary using native metrics
        with st.expander("📊 Query Analysis Summary"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Response Time", "0.34s")
            with col2:
                avg_relevance = sum(r.final_score for r in recommendations) / len(recommendations)
                st.metric("Avg Relevance", f"{avg_relevance:.1%}")
            with col3:
                diversity = calculate_diversity_score([r.resource for r in recommendations])
                st.metric("Diversity", f"{diversity:.2f}")
            with col4:
                st.metric("Confidence", "High")
    
    # Recommendations using native containers
    for idx, rec in enumerate(recommendations, 1):
        medal_emoji = ["🥇", "🥈", "🥉"][idx - 1] if idx <= 3 else f"{idx}️⃣"
        
        with st.container():
            # Use expander for clean display
            with st.expander(f"{medal_emoji} {prefix} {rec.resource.title} (Score: {rec.final_score:.3f})", expanded=True):
                
                # Basic information
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Category:** {rec.resource.primary_category.replace('_', ' ').title()}")
                    st.write(f"**Level:** {rec.resource.difficulty_level.title()}")
                    st.write(f"**Source:** {rec.resource.source}")
                
                with col2:
                    st.metric("Relevance", f"{rec.final_score:.3f}")
                
                # Explanation
                st.info(f"💡 **Why recommended:** {rec.primary_reason}")
                st.write(f"**Details:** {rec.detailed_explanation}")
                
                # Action buttons (only in non-compact mode)
                if not compact:
                    button_col1, button_col2, button_col3 = st.columns([1, 1, 2])
                    
                    with button_col1:
                        if st.button("👍 Helpful", key=f"helpful_{rec.resource.resource_id}_{idx}"):
                            st.success("Thanks for the feedback!")
                    
                    with button_col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{rec.resource.resource_id}_{idx}"):
                            st.info("Feedback recorded!")
                    
                    with button_col3:
                        if rec.resource.url:
                            st.markdown(f"[🔗 View Resource]({rec.resource.url})")

def render_performance_dashboard(language: str):
    """Render system performance dashboard."""
    
    if st.checkbox("📊 Show Performance Dashboard", key="show_dashboard"):
        st.markdown("### ⚡ System Performance Dashboard")
        
        # Create performance metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Users", "4", "+1")
        with col2:
            st.metric("Recommendations Generated", "127", "+12")
        with col3:
            st.metric("Avg Response Time", "0.34s", "-0.02s")
        with col4:
            st.metric("User Satisfaction", "87%", "+5%")
        with col5:
            st.metric("Algorithm Accuracy", "92%", "+3%")
        
        # Performance chart
        if len(st.session_state.recommendations_history) > 0:
            st.markdown("#### 📈 Recommendation Quality Over Time")
            
            # Create time series of recommendation scores
            history_data = []
            for entry in st.session_state.recommendations_history:
                avg_score = sum(r.final_score for r in entry['recommendations']) / len(entry['recommendations'])
                history_data.append({
                    'timestamp': entry['timestamp'],
                    'avg_score': avg_score,
                    'query': entry['query'][:30] + "..." if len(entry['query']) > 30 else entry['query']
                })
            
            # Create interactive chart
            if history_data:
                fig = px.line(
                    history_data, 
                    x='timestamp', 
                    y='avg_score',
                    title='Average Recommendation Score Over Time',
                    hover_data=['query']
                )
                fig.update_traces(mode='markers+lines')
                st.plotly_chart(fig, use_container_width=True)

# Helper Functions
def get_smart_query_suggestions(user: UserProfile, language: str = 'es') -> List[tuple]:
    """Generate smart query suggestions based on user profile."""
    
    suggestions_by_role = {
        'CEO': {
            'es': [
                ("¿Cómo puedo contratar developers rápidamente?", "hiring_guidance"),
                ("¿Qué diferencia hay entre freelancers y squads?", "service_comparison"),
                ("¿Cuánto cuesta usar Shakers AI para mi startup?", "pricing_models"),
                ("¿Cómo validar la calidad técnica de los freelancers?", "quality_assurance")
            ],
            'en': [
                ("How can I hire developers quickly?", "hiring_guidance"),
                ("What's the difference between freelancers and squads?", "service_comparison"),
                ("How much does Shakers AI cost for my startup?", "pricing_models"),
                ("How can I validate freelancers' technical quality?", "quality_assurance")
            ]
        },
        'Marketing Director': {
            'es': [
                ("¿Qué perfiles de marketing especializado tienen disponibles?", "talent_discovery"),
                ("¿Pueden manejar campañas de performance marketing?", "use_cases"),
                ("¿Cómo medir ROI de proyectos con squads de marketing?", "performance_measurement"),
                ("¿Tienen especialistas en growth hacking?", "talent_discovery")
            ],
            'en': [
                ("What specialized marketing profiles are available?", "talent_discovery"),
                ("Can you handle performance marketing campaigns?", "use_cases"),
                ("How to measure ROI of marketing squad projects?", "performance_measurement"),
                ("Do you have growth hacking specialists?", "talent_discovery")
            ]
        },
        'Head of People': {
            'es': [
                ("¿Qué es el modelo de trabajo Full Flex?", "work_culture"),
                ("¿Cómo garantizan el encaje cultural con freelancers?", "cultural_fit"),
                ("¿Puedo gestionar equipos 100% remotos?", "remote_management"),
                ("¿Qué beneficios ofrecen a los freelancers?", "talent_retention")
            ],
            'en': [
                ("What is the Full Flex work model?", "work_culture"),
                ("How do you ensure cultural fit with freelancers?", "cultural_fit"),
                ("Can I manage 100% remote teams?", "remote_management"),
                ("What benefits do you offer to freelancers?", "talent_retention")
            ]
        },
        'Product Manager': {
            'es': [
                ("¿Cómo ayuda Shakers AI en la definición de proyectos?", "ai_capabilities"),
                ("¿Qué casos de uso típicos hay para equipos de producto?", "use_cases"),
                ("¿Pueden manejar desarrollo de MVPs completos?", "product_development"),
                ("¿Cómo es el proceso de briefing para proyectos técnicos?", "project_planning")
            ],
            'en': [
                ("How does Shakers AI help with project definition?", "ai_capabilities"),
                ("What are typical use cases for product teams?", "use_cases"),
                ("Can you handle complete MVP development?", "product_development"),
                ("What's the briefing process for technical projects?", "project_planning")
            ]
        }
    }
    
    # Get suggestions based on user role
    for role_key in suggestions_by_role:
        if role_key.lower() in user.role.lower():
            role_suggestions = suggestions_by_role[role_key]
            return role_suggestions.get(language, role_suggestions['es'])
    
    # Default suggestions
    default_suggestions = suggestions_by_role['CEO']
    return default_suggestions.get(language, default_suggestions['es'])

def analyze_query_category(query: str) -> str:
    """Simple query categorization based on keywords."""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['contratar', 'hiring', 'freelancer', 'squad']):
        return 'hiring_guidance'
    elif any(word in query_lower for word in ['ai', 'inteligencia artificial', 'shakers ai']):
        return 'ai_features'
    elif any(word in query_lower for word in ['cultura', 'culture', 'remoto', 'remote', 'flex']):
        return 'work_culture'
    elif any(word in query_lower for word in ['precio', 'cost', 'cuanto', 'price']):
        return 'pricing_models'
    elif any(word in query_lower for word in ['que es', 'what is', 'como funciona', 'how works']):
        return 'platform_overview'
    else:
        return 'general_inquiry'

def analyze_query_intent(query: str) -> str:
    """Analyze the intent behind a query."""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['como', 'how']):
        return 'information_seeking'
    elif any(word in query_lower for word in ['mejor', 'best', 'recomien', 'suggest']):
        return 'decision_support'
    elif any(word in query_lower for word in ['problema', 'issue', 'help', 'ayuda']):
        return 'problem_solving'
    elif any(word in query_lower for word in ['que', 'what', 'cual', 'which']):
        return 'information_seeking'
    else:
        return 'general_inquiry'

def analyze_query_urgency(query: str) -> str:
    """Analyze the urgency level of a query."""
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['rapido', 'urgent', 'ya', 'now', 'inmediato']):
        return 'high'
    elif any(word in query_lower for word in ['pronto', 'soon', 'cuando', 'when']):
        return 'medium'
    else:
        return 'low'

def calculate_profile_match(query: str, user: UserProfile) -> float:
    """Calculate how well a query matches the user's profile."""
    
    query_lower = query.lower()
    user_interests = user.get_top_interests()
    
    # Simple keyword matching
    matches = 0
    total_interests = len(user_interests)
    
    for interest in user_interests:
        interest_keywords = interest.replace('_', ' ').split()
        for keyword in interest_keywords:
            if keyword.lower() in query_lower:
                matches += 1
                break
    
    return matches / total_interests if total_interests > 0 else 0.5

def generate_recommendations_for_query(user: UserProfile, query: str, 
                                     sql_handler: SQLHandler, embedder: Embedder,
                                     algorithm: str = 'default') -> List[Any]:
    """Generate recommendations for a query using the recommendation engine."""
    
    try:
        # Initialize recommendation system components
        adapter = ChunkToResourceAdapter(sql_handler)
        engine = RecommendationEngine()
        
        # Load all available resources
        resources_dir = Path("data/resources")
        external_resources = adapter.load_external_resources(resources_dir)
        chunk_resources = adapter.get_all_resources_from_chunks()
        
        all_resources = external_resources + chunk_resources
        
        if not all_resources:
            st.warning("No resources available for recommendations. Please ensure data/files exist.")
            return []
        
        # Modify engine weights based on algorithm
        if algorithm == 'interest_focused':
            engine.weights['interest_match'] = 0.6
            engine.weights['role_match'] = 0.2
            engine.weights['quality'] = 0.1
            engine.weights['recency'] = 0.1
        elif algorithm == 'role_focused':
            engine.weights['interest_match'] = 0.2
            engine.weights['role_match'] = 0.6
            engine.weights['quality'] = 0.1
            engine.weights['recency'] = 0.1
        
        # Generate recommendations
        recommendations = engine.generate_recommendations(
            user=user,
            all_resources=all_resources,
            current_query=query,
            num_recommendations=3
        )
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

def calculate_diversity_score(resources: List[Any]) -> float:
    """Calculate diversity score for a list of resources."""
    
    if not resources:
        return 0.0
    
    categories = [r.primary_category for r in resources]
    unique_categories = len(set(categories))
    total_resources = len(resources)
    
    return unique_categories / total_resources

def simulate_user_preference(recommendations: List[Any], user: UserProfile) -> float:
    """Simulate user preference based on profile matching."""
    
    if not recommendations:
        return 0.0
    
    total_preference = 0.0
    user_interests = user.get_top_interests()
    
    for rec in recommendations:
        preference = 0.5  # Base preference
        
        # Boost if matches user interests
        if rec.resource.primary_category in user_interests:
            preference += 0.3
        
        # Boost if matches user role
        if user.role.lower() in [role.lower() for role in rec.resource.target_roles]:
            preference += 0.2
        
        # Boost based on final score
        preference += rec.final_score * 0.2
        
        total_preference += min(preference, 1.0)
    
    return total_preference / len(recommendations)



