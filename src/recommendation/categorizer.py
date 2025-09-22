"""
Simple content categorization for recommendation system.

Categorizes content based on keyword matching and rules.
This is a simple implementation - could be enhanced with ML models later.
"""

class ContentCategorizer:
    """
    Simple rule-based categorizer for content.
    
    Uses keyword matching to assign primary categories to content.
    Categories align with user interests and Shakers platform topics.
    """
    
    def __init__(self):
        # Keywords for each category (in Spanish and English)
        self.category_keywords = {
            "platform_basics": [
                "shakers", "plataforma", "platform", "qué es", "what is", 
                "funciona", "works", "overview", "introducción", "getting started"
            ],
            "hiring_guidance": [
                "contratar", "hiring", "contratación", "reclutamiento", "recruitment",
                "proceso", "process", "entrevista", "interview", "selección", "brief"
            ],
            "ai_features": [
                "ia", "ai", "inteligencia artificial", "artificial intelligence",
                "shakers ai", "copiloto", "copilot", "automatización", "automation"
            ],
            "team_building": [
                "squad", "equipo", "team", "multidisciplinar", "coordinación",
                "collaboration", "colaboración", "grupo", "group"
            ],
            "work_culture": [
                "cultura", "culture", "full flex", "flexible", "remoto", "remote",
                "beneficios", "benefits", "oficina", "office", "trabajo", "work"
            ],
            "business_intelligence": [
                "métricas", "metrics", "serie a", "financiación", "funding",
                "inversión", "investment", "tracción", "traction", "crecimiento"
            ],
            "legal_contracts": [
                "contrato", "contract", "legal", "facturación", "billing",
                "confidencialidad", "confidentiality", "propiedad", "intellectual"
            ],
            "use_cases": [
                "caso", "case", "ejemplo", "example", "proyecto", "project",
                "mvp", "marketing", "desarrollo", "development"
            ],
            "technical_integration": [
                "técnico", "technical", "desarrollo", "development", "código", "code",
                "api", "integración", "integration", "developer"
            ],
            "talent_management": [
                "talento", "talent", "freelancer", "gestión", "management",
                "recursos humanos", "hr", "people", "personas"
            ]
        }
    
    def categorize_content(self, summary: str, content: str = "") -> str:
        """
        Categorize content based on summary and content text.
        
        Args:
            summary: Main summary text to analyze
            content: Full content text (optional)
            
        Returns:
            Primary category string
        """
        text_to_analyze = (summary + " " + content).lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    score += 1
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        # Default category if no matches
        return "platform_basics"
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description of category"""
        descriptions = {
            "platform_basics": "Basic information about Shakers platform",
            "hiring_guidance": "How to hire and work with freelancers",
            "ai_features": "AI-powered features and automation",
            "team_building": "Building and managing teams",
            "work_culture": "Work flexibility and company culture", 
            "business_intelligence": "Business metrics and market insights",
            "legal_contracts": "Legal and contractual aspects",
            "use_cases": "Practical examples and case studies",
            "technical_integration": "Technical implementation details",
            "talent_management": "Managing freelance talent"
        }
        return descriptions.get(category, "General information")