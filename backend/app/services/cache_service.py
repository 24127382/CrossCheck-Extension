"""Cache service - stores fact-check results for later use in explanation"""
from typing import Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from ..schemas.evidence import Evidence

@dataclass
class CachedFactCheckResult:
    """Cached result from a fact-check request"""
    claim: str
    verdict: str
    confidence: float
    evidences: List[Evidence]
    timestamp: float
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry has expired (default: 1 hour)"""
        return datetime.fromtimestamp(self.timestamp) < datetime.now() - timedelta(seconds=ttl_seconds)

class CacheService:
    """In-memory cache for fact-check results"""
    
    def __init__(self, max_cache_size: int = 100):
        # Simple dict-based cache: claim -> CachedFactCheckResult
        self.cache: dict[str, CachedFactCheckResult] = {}
        self.max_cache_size = max_cache_size
    
    def save(self, claim: str, verdict: str, confidence: float, evidences: List[Evidence]) -> None:
        """Save fact-check result to cache"""
        # Simple FIFO eviction if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            del self.cache[oldest_key]
            print(f"[Cache] Evicted oldest entry: {oldest_key[:50]}")
        
        self.cache[claim] = CachedFactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidences=evidences,
            timestamp=datetime.now().timestamp()
        )
        print(f"[Cache] ✅ Saved result for claim: {claim[:50]}")
    
    def get(self, claim: str, ttl_seconds: int = 3600) -> Optional[CachedFactCheckResult]:
        """Retrieve fact-check result from cache"""
        if claim not in self.cache:
            print(f"[Cache] ⚠️ Cache miss for claim: {claim[:50]}")
            return None
        
        result = self.cache[claim]
        
        if result.is_expired(ttl_seconds):
            print(f"[Cache] ⚠️ Cache expired for claim: {claim[:50]}")
            del self.cache[claim]
            return None
        
        print(f"[Cache] ✅ Cache hit for claim: {claim[:50]}")
        return result
    
    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        print("[Cache] ✅ Cache cleared")
    
    def stats(self) -> dict:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_cache_size,
            "entries": [
                {
                    "claim": k[:50],
                    "verdict": v.verdict,
                    "age_seconds": int(datetime.now().timestamp() - v.timestamp)
                }
                for k, v in self.cache.items()
            ]
        }

# Global cache instance
cache_service = CacheService()
