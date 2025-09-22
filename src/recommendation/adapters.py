import json
from pathlib import Path
from typing import List, Dict, Any
from database.sql_handler import SQLHandler
from .models import Resource
from .categorizer import ContentCategorizer

class ChunkToResourceAdapter:
    """
    Converts chunks from the existing RAG system into Resource objects
    for the recommendation system.
    
    This allows reusing all existing processed content without changes
    to the current document processing pipeline.
    """
    
    def __init__(self, sql_handler: SQLHandler):
        self.sql_handler = sql_handler
        self.categorizer = ContentCategorizer()
        
    def convert_chunk_to_resource(self, chunk: Dict[str, Any]) -> Resource:
        """
        Convert a single chunk from SQL database to Resource format.
        
        Args:
            chunk: Dictionary from sql_handler.get_all_chunks()
            
        Returns:
            Resource object ready for recommendation system
        """
        # Generate title from summary (first 50 chars + ...)
        title = self._generate_title_from_summary(chunk.get('summary', ''))
        
        # Categorize content based on summary/content
        primary_category = self.categorizer.categorize_content(
            chunk.get('summary', ''),
            chunk.get('content', '')
        )
        
        # Infer target roles based on category
        target_roles = self._infer_target_roles(primary_category)
        
        return Resource(
            resource_id=f"chunk_{chunk['id']}",
            title=title,
            summary=chunk['summary'] or "Content from processed document",
            content_type=chunk['chunk_type'],
            source="internal_chunk", 
            language="es",  # Default from your knowledge base
            primary_category=primary_category,
            secondary_categories=[],
            target_roles=target_roles,
            difficulty_level="intermediate",
            tags=self._generate_tags(chunk),
            url=None,
            popularity_score=0.8,  # Internal content is high quality
            quality_score=0.9
        )
    
    def get_all_resources_from_chunks(self) -> List[Resource]:
        """
        Get all chunks from database and convert them to Resources.
        
        Returns:
            List of Resource objects from all processed chunks
        """
        try:
            chunks = self.sql_handler.get_all_chunks()
            resources = []
            
            for chunk in chunks:
                try:
                    resource = self.convert_chunk_to_resource(chunk)
                    resources.append(resource)
                except Exception as e:
                    print(f"Warning: Failed to convert chunk {chunk.get('id', 'unknown')}: {e}")
                    continue
            
            print(f"Successfully converted {len(resources)} chunks to resources")
            return resources
            
        except Exception as e:
            print(f"Error accessing chunk database: {e}")
            return []
    
    def load_external_resources(self, resources_dir: Path) -> List[Resource]:
        """
        Load external resources from JSON files.
        
        Args:
            resources_dir: Path to directory containing resource JSON files
            
        Returns:
            List of Resource objects from external files
        """
        resources = []
        
        # Load external articles
        external_articles_file = resources_dir / "external_articles.json"
        if external_articles_file.exists():
            resources.extend(self._load_resources_from_file(external_articles_file))
        
        # Load tutorials
        tutorials_file = resources_dir / "tutorials.json"
        if tutorials_file.exists():
            resources.extend(self._load_resources_from_file(tutorials_file))
        
        # Load generated resources
        generated_file = resources_dir / "generated_resources.json"
        if generated_file.exists():
            resources.extend(self._load_resources_from_file(generated_file))
        
        return resources
    
    def _load_resources_from_file(self, file_path: Path) -> List[Resource]:
        """Load resources from a single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            resources = []
            for item in data:
                try:
                    resource = Resource.from_json_dict(item)
                    resources.append(resource)
                except Exception as e:
                    print(f"Warning: Failed to load resource from {file_path}: {e}")
                    continue
            
            print(f"Loaded {len(resources)} resources from {file_path.name}")
            return resources
            
        except Exception as e:
            print(f"Error loading resources from {file_path}: {e}")
            return []
    
    def _generate_title_from_summary(self, summary: str) -> str:
        """Generate a short title from summary"""
        if not summary:
            return "Untitled Resource"
        
        # Take first sentence or first 50 characters
        first_sentence = summary.split('.')[0]
        if len(first_sentence) <= 50:
            return first_sentence
        else:
            return summary[:47] + "..."
    
    def _generate_tags(self, chunk: Dict[str, Any]) -> List[str]:
        """Generate simple tags based on chunk content"""
        tags = []
        summary = chunk.get('summary', '').lower()
        
        # Simple keyword-based tagging
        if 'shakers' in summary:
            tags.append('shakers')
        if 'ai' in summary or 'inteligencia artificial' in summary:
            tags.append('ai')
        if 'squad' in summary or 'equipo' in summary:
            tags.append('teams')
        if 'freelancer' in summary:
            tags.append('freelancing')
        if 'contract' in summary or 'contrat' in summary:
            tags.append('hiring')
        
        return tags
    
    def _infer_target_roles(self, category: str) -> List[str]:
        """Infer target roles based on content category"""
        role_mapping = {
            "platform_basics": ["founder", "hr_manager", "product_manager"],
            "hiring_guidance": ["founder", "hr_manager"],
            "ai_features": ["founder", "cto", "product_manager"],
            "team_building": ["founder", "product_manager", "cto"],
            "work_culture": ["hr_manager", "founder"],
            "business_intelligence": ["founder", "product_manager"],
            "legal_contracts": ["founder", "hr_manager"],
            "use_cases": ["product_manager", "marketing_director"],
            "technical_integration": ["cto", "developer"],
            "talent_management": ["hr_manager", "marketing_director"]
        }
        
        return role_mapping.get(category, ["founder", "product_manager"])