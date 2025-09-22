"""
Explanation generator for recommendations.

Creates human-readable explanations for why each resource
is recommended to a specific user.
"""

from typing import Dict, Any
from .models import UserProfile, Resource

class RecommendationExplainer:
    """
    Generates explanations for why resources are recommended to users.
    
    Creates both primary reasons and detailed explanations
    that users can understand and trust.
    """
    
    def generate_explanation(self, 
                           user: UserProfile, 
                           resource: Resource,
                           current_query: str = "") -> Dict[str, str]:
        """
        Generate explanation for why a resource is recommended.
        
        Args:
            user: UserProfile of the user
            resource: Resource being recommended
            current_query: Current query context (optional)
            
        Returns:
            Dictionary with explanation components
        """
        # Determine primary reason for recommendation
        primary_reason = self._determine_primary_reason(user, resource, current_query)
        
        # Generate detailed explanation
        detailed_explanation = self._generate_detailed_explanation(
            user, resource, current_query, primary_reason
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(user, resource)
        
        return {
            'primary_reason': primary_reason,
            'detailed_explanation': detailed_explanation,
            'confidence_level': confidence_level
        }
    
    def _determine_primary_reason(self, 
                                user: UserProfile, 
                                resource: Resource, 
                                query: str) -> str:
        """Determine the main reason for this recommendation"""
        
        # Check for direct interest match
        user_interests = user.get_top_interests()
        if resource.primary_category in user_interests:
            return f"Based on your interest in {resource.primary_category.replace('_', ' ')}"
        
        # Check for role match
        if user.role.lower() in [role.lower() for role in resource.target_roles]:
            return f"Relevant for your role as {user.role}"
        
        # Check for query relevance
        if query and self._has_query_relevance(query, resource):
            return "Related to your current question"
        
        # Check for recent activity
        recent_categories = user.get_recent_categories(3)
        if resource.primary_category in recent_categories:
            return "Based on your recent queries"
        
        # Default reason
        return "High-quality content for your profile"
    
    def _generate_detailed_explanation(self, 
                                     user: UserProfile,
                                     resource: Resource,
                                     query: str,
                                     primary_reason: str) -> str:
        """Generate detailed explanation combining multiple factors"""
        
        explanation_parts = []
        
        # Add primary reason detail
        explanation_parts.append(primary_reason + ".")
        
        # Add role relevance if applicable
        if user.role.lower() in [role.lower() for role in resource.target_roles]:
            explanation_parts.append(
                f"This {resource.content_type} is specifically designed for {user.role}s."
            )
        
        # Add content type and source info
        if resource.source == "external_article":
            explanation_parts.append("This is a high-quality external article.")
        elif resource.source == "tutorial":
            explanation_parts.append("This tutorial provides step-by-step guidance.")
        elif resource.source == "internal_chunk":
            explanation_parts.append("This information comes from our curated knowledge base.")
        
        # Add difficulty level context
        if resource.difficulty_level == user.experience_level:
            explanation_parts.append(f"The content matches your {user.experience_level} level.")
        elif resource.difficulty_level == "beginner" and user.experience_level in ["mid", "senior"]:
            explanation_parts.append("This foundational content might provide useful background.")
        
        # Add query connection if relevant
        if query and self._has_query_relevance(query, resource):
            explanation_parts.append("It directly addresses concepts from your question.")
        
        return " ".join(explanation_parts)
    
    def _has_query_relevance(self, query: str, resource: Resource) -> bool:
        """Check if resource is relevant to the query"""
        query_lower = query.lower()
        resource_text = (resource.title + " " + resource.summary + " " + " ".join(resource.tags)).lower()
        
        # Simple keyword matching
        query_words = [word for word in query_lower.split() if len(word) > 2]
        matches = sum(1 for word in query_words if word in resource_text)
        
        return matches > 0 and len(query_words) > 0
    
    def _determine_confidence_level(self, user: UserProfile, resource: Resource) -> str:
        """Determine confidence level for the recommendation"""
        score = 0
        
        # High confidence factors
        user_interests = user.get_top_interests()
        if resource.primary_category in user_interests:
            score += 2
        
        if user.role.lower() in [role.lower() for role in resource.target_roles]:
            score += 2
        
        # Medium confidence factors
        if resource.quality_score > 0.8:
            score += 1
        
        recent_categories = user.get_recent_categories(3)
        if resource.primary_category in recent_categories:
            score += 1
        
        # Determine level
        if score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        else:
            return "low"

    def get_explanation_templates(self, language: str = "en") -> Dict[str, str]:
        """Get explanation templates for different languages"""
        if language == "es":
            return {
                "interest_match": "Basado en tu interés en {category}",
                "role_match": "Relevante para tu rol como {role}",
                "query_match": "Relacionado con tu pregunta actual",
                "recent_activity": "Basado en tus consultas recientes",
                "high_quality": "Contenido de alta calidad para tu perfil"
            }
        else:  # English default
            return {
                "interest_match": "Based on your interest in {category}",
                "role_match": "Relevant for your role as {role}",
                "query_match": "Related to your current question",
                "recent_activity": "Based on your recent queries",
                "high_quality": "High-quality content for your profile"
            }
        