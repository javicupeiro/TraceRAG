"""
Data models for the recommendation system.

Simple dataclasses that represent:
- UserProfile: User information and computed interests
- Resource: Content that can be recommended
- Recommendation: A single recommendation with score and explanation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

@dataclass
class UserProfile:
    """
    Represents a user profile with query history and computed interests.
    
    This class handles loading user data from JSON files and computing
    interest scores based on query history.
    """
    user_id: str
    name: str
    role: str
    company_type: str
    experience_level: str
    query_history: List[Dict[str, Any]]
    computed_interests: Dict[str, float] = field(default_factory=dict)
    language_preference: str = "en"
    
    @classmethod
    def load_from_file(cls, user_file_path: Path) -> 'UserProfile':
        """Load user profile from JSON file"""
        with open(user_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profile = data['profile']
        query_history = data['query_history']
        preferences = data.get('preferences', {})
        computed_profile = data.get('computed_profile', {})
        
        return cls(
            user_id=data['user_id'],
            name=profile['name'],
            role=profile['role'],
            company_type=profile['company_type'],
            experience_level=profile['experience_level'],
            query_history=query_history,
            computed_interests=computed_profile.get('interest_scores', {}),
            language_preference=preferences.get('language', 'en')
        )
    
    def get_recent_categories(self, limit: int = 5) -> List[str]:
        """Get categories from most recent queries"""
        recent_queries = sorted(
            self.query_history, 
            key=lambda x: x['timestamp'], 
            reverse=True
        )[:limit]
        return [q['category'] for q in recent_queries]
    
    def get_top_interests(self, limit: int = 3) -> List[str]:
        """Get top interest categories by score"""
        if not self.computed_interests:
            return self.get_recent_categories(limit)
        
        sorted_interests = sorted(
            self.computed_interests.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [interest[0] for interest in sorted_interests[:limit]]

@dataclass
class Resource:
    """
    Represents a resource that can be recommended.
    
    Can be either internal content (chunks) or external resources
    (articles, tutorials).
    """
    resource_id: str
    title: str
    summary: str
    content_type: str
    source: str
    language: str
    primary_category: str
    secondary_categories: List[str] = field(default_factory=list)
    target_roles: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    popularity_score: float = 0.5
    quality_score: float = 0.5
    
    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Create Resource from JSON dictionary"""
        content = data['content']
        categorization = data['categorization']
        metadata = data['metadata']
        rec_metadata = data.get('recommendation_metadata', {})
        
        return cls(
            resource_id=data['resource_id'],
            title=content['title'],
            summary=content['summary'],
            content_type=content['type'],
            source=metadata['source'],
            language=metadata['language'],
            primary_category=categorization['primary_category'],
            secondary_categories=categorization.get('secondary_categories', []),
            target_roles=categorization.get('target_roles', []),
            difficulty_level=categorization.get('difficulty_level', 'intermediate'),
            tags=categorization.get('tags', []),
            url=content.get('url'),
            popularity_score=rec_metadata.get('popularity_score', 0.5),
            quality_score=rec_metadata.get('quality_score', 0.5)
        )

@dataclass
class Recommendation:
    """
    Represents a single recommendation with scores and explanation.
    """
    resource: Resource
    relevance_score: float
    diversity_score: float
    final_score: float
    rank: int
    primary_reason: str
    detailed_explanation: str
    confidence_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary for JSON serialization"""
        return {
            "resource_id": self.resource.resource_id,
            "rank": self.rank,
            "scores": {
                "relevance_score": self.relevance_score,
                "diversity_score": self.diversity_score,
                "final_score": self.final_score
            },
            "explanation": {
                "primary_reason": self.primary_reason,
                "detailed_explanation": self.detailed_explanation,
                "confidence_level": self.confidence_level
            },
            "resource_info": {
                "title": self.resource.title,
                "summary": self.resource.summary,
                "url": self.resource.url,
                "category": self.resource.primary_category
            }
        }