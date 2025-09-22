"""
Main recommendation engine implementing the core algorithm.

Simple implementation focusing on:
- User interest matching
- Content relevance scoring  
- Basic diversity ensuring
- Top 3 recommendations output
"""

import random
from typing import List, Dict, Any
from .models import UserProfile, Resource, Recommendation
from .explainer import RecommendationExplainer

class RecommendationEngine:
    """
    Core recommendation engine that generates personalized suggestions.
    
    Algorithm:
    1. Score resources based on user interests and query context
    2. Apply diversity filtering to avoid all same-category results
    3. Generate explanations for top recommendations
    4. Return ranked list of 2-3 recommendations
    """
    
    def __init__(self):
        self.explainer = RecommendationExplainer()
        
        # Weights for final scoring
        self.weights = {
            'interest_match': 0.4,    # User's historical interests
            'role_match': 0.3,        # Target role alignment
            'quality': 0.2,           # Resource quality score
            'recency': 0.1           # Content freshness
        }
    
    def generate_recommendations(self, 
                               user: UserProfile,
                               all_resources: List[Resource],
                               current_query: str = "",
                               num_recommendations: int = 3) -> List[Recommendation]:
        """
        Generate personalized recommendations for a user.
        
        Args:
            user: UserProfile with interests and history
            all_resources: All available resources to recommend from
            current_query: Current user query for context (optional)
            num_recommendations: Number of recommendations to return (default 3)
            
        Returns:
            List of Recommendation objects, ranked by relevance
        """
        # Filter resources by language preference
        language_filtered = self._filter_by_language(all_resources, user.language_preference)
        
        # Score all resources
        scored_resources = []
        for resource in language_filtered:
            score = self._calculate_resource_score(user, resource, current_query)
            scored_resources.append((resource, score))
        
        # Sort by score
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering
        diverse_resources = self._ensure_diversity(scored_resources, num_recommendations)
        
        # Generate recommendations with explanations
        recommendations = []
        for rank, (resource, score) in enumerate(diverse_resources, 1):
            explanation = self.explainer.generate_explanation(user, resource, current_query)
            
            recommendation = Recommendation(
                resource=resource,
                relevance_score=score,
                diversity_score=self._calculate_diversity_score(resource, diverse_resources),
                final_score=score,
                rank=rank,
                primary_reason=explanation['primary_reason'],
                detailed_explanation=explanation['detailed_explanation'],
                confidence_level=explanation['confidence_level']
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _filter_by_language(self, resources: List[Resource], preferred_language: str) -> List[Resource]:
        """Filter resources by user's language preference"""
        filtered = [r for r in resources if r.language == preferred_language]
        
        # If no resources in preferred language, include all
        if not filtered:
            print(f"Warning: No resources found for language '{preferred_language}', including all languages")
            return resources
        
        return filtered
    
    def _calculate_resource_score(self, user: UserProfile, resource: Resource, query: str) -> float:
        """
        Calculate relevance score for a resource given user profile.
        
        Combines multiple factors:
        - Interest match: How well resource matches user's interests
        - Role match: How well resource targets user's role
        - Quality: Intrinsic quality of the resource
        - Query relevance: How well it matches current query
        """
        scores = {
            'interest_match': self._score_interest_match(user, resource),
            'role_match': self._score_role_match(user, resource),
            'quality': resource.quality_score,
            'recency': self._score_recency(resource)
        }
        
        # Add query relevance if query provided
        if query:
            scores['query_relevance'] = self._score_query_relevance(query, resource)
            self.weights['query_relevance'] = 0.2
            # Rebalance other weights
            for key in ['interest_match', 'role_match', 'quality', 'recency']:
                self.weights[key] *= 0.8
        
        # Calculate weighted final score
        final_score = sum(scores[key] * self.weights.get(key, 0) for key in scores)
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _score_interest_match(self, user: UserProfile, resource: Resource) -> float:
        """Score how well resource matches user interests"""
        user_interests = user.get_top_interests()
        
        # Direct category match
        if resource.primary_category in user_interests:
            return 1.0
        
        # Secondary category match
        for secondary in resource.secondary_categories:
            if secondary in user_interests:
                return 0.7
        
        # Recent query category match
        recent_categories = user.get_recent_categories(3)
        if resource.primary_category in recent_categories:
            return 0.8
        
        return 0.3  # Base score
    
    def _score_role_match(self, user: UserProfile, resource: Resource) -> float:
        """Score how well resource targets user's role"""
        # Exact role match
        if user.role.lower() in [role.lower() for role in resource.target_roles]:
            return 1.0
        
        # Partial role matching (e.g., "CEO" matches "founder")
        role_synonyms = {
            'ceo': 'founder',
            'founder': 'ceo',
            'head of people': 'hr_manager',
            'marketing director': 'marketing_director'
        }
        
        user_role_normalized = user.role.lower()
        if user_role_normalized in role_synonyms:
            synonym_role = role_synonyms[user_role_normalized]
            if synonym_role in resource.target_roles:
                return 0.9
        
        # General match for common roles
        if 'founder' in resource.target_roles and user.company_type in ['startup', 'scale_up']:
            return 0.7
        
        return 0.5  # Neutral score
    
    def _score_recency(self, resource: Resource) -> float:
        """Score based on content freshness (placeholder)"""
        # For now, external sources get higher recency
        if resource.source == "external_article":
            return 0.8
        elif resource.source == "tutorial":
            return 0.7
        else:  # internal chunks
            return 0.9  # Internal content is always fresh
    
    def _score_query_relevance(self, query: str, resource: Resource) -> float:
        """Score relevance to current query (simple keyword matching)"""
        query_lower = query.lower()
        text_to_check = (resource.title + " " + resource.summary + " " + " ".join(resource.tags)).lower()
        
        # Count word matches
        query_words = query_lower.split()
        matches = sum(1 for word in query_words if len(word) > 2 and word in text_to_check)
        
        if len(query_words) == 0:
            return 0.5
        
        return min(matches / len(query_words), 1.0)
    
    def _ensure_diversity(self, scored_resources: List[tuple], num_recommendations: int) -> List[tuple]:
        """
        Ensure diversity in recommendations by avoiding all same-category results.
        
        Args:
            scored_resources: List of (resource, score) tuples, sorted by score
            num_recommendations: Target number of recommendations
            
        Returns:
            Filtered list ensuring category diversity
        """
        if len(scored_resources) <= num_recommendations:
            return scored_resources[:num_recommendations]
        
        selected = []
        used_categories = set()
        
        # First pass: select top resources from different categories
        for resource, score in scored_resources:
            if len(selected) >= num_recommendations:
                break
                
            if resource.primary_category not in used_categories:
                selected.append((resource, score))
                used_categories.add(resource.primary_category)
        
        # Second pass: fill remaining slots with best remaining resources
        if len(selected) < num_recommendations:
            for resource, score in scored_resources:
                if len(selected) >= num_recommendations:
                    break
                    
                if (resource, score) not in selected:
                    selected.append((resource, score))
        
        return selected[:num_recommendations]
    
    def _calculate_diversity_score(self, resource: Resource, all_selected: List[tuple]) -> float:
        """Calculate diversity score for a resource within the selection"""
        categories = [r[0].primary_category for r in all_selected]
        unique_categories = len(set(categories))
        total_resources = len(all_selected)
        
        return unique_categories / total_resources if total_resources > 0 else 1.0