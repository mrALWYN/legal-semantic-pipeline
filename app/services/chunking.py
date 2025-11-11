import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass
class SemanticUnit:
    """Single unit of text with classification."""
    text: str
    index: int
    category: str
    confidence: float


class LegalSemanticClassifier:
    """
    Rule-based + pattern-based classifier for Indian legal judgments.
    Classifies sentences into 8 semantic categories.
    """
    
    def __init__(self):
        # Category definitions with weighted patterns
        self.category_patterns = {
            'Facts': {
                'strong': [
                    r'\b(?:brief facts?|factual (?:matrix|background)|facts of (?:the )?case)\b',
                    r'\b(?:incident|occurred|took place|happened) on\b',
                    r'\b(?:filed|instituted|lodged) (?:a |an )?(?:complaint|fir|petition|suit|appeal)\b',
                    r'\balleged(?:ly)? that\b',
                    r'\bfacts (?:leading to|arising out of|giving rise to)\b',
                    r'\b(?:petitioner|appellant|plaintiff|complainant) (?:is|was|has been)\b',
                ],
                'medium': [
                    r'\bbackground\b',
                    r'\bchronology\b',
                    r'\bevents (?:that )?(?:led to|preceded)\b',
                    r'\bcircumstances\b',
                    r'\bsequence of events\b',
                ],
                'weak': [
                    r'\bon \d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                    r'\bagainst (?:the |this )?(?:order|judgment|decision)\b',
                ]
            },
            
            'Arguments_Petitioner': {
                'strong': [
                    r'\b(?:learned )?(?:counsel|advocate|senior advocate) (?:for|appearing for|representing) (?:the )?(?:petitioner|appellant|plaintiff)\b',
                    r'\bpetitioner(?:\'s)? (?:contends?|submits?|argues?|pleads?|claims?)\b',
                    r'\bappellant(?:\'s)? (?:contends?|submits?|argues?|pleads?|claims?)\b',
                    r'\b(?:it is|it was) (?:contended|submitted|argued|urged|pleaded) (?:by|on behalf of) (?:the )?(?:petitioner|appellant)\b',
                    r'\blearned counsel for (?:the )?(?:petitioner|appellant) (?:has )?(?:contended|submitted|argued|urged)\b',
                ],
                'medium': [
                    r'\bpetitioner(?:\'s)? (?:case|stand|position|contention)\b',
                    r'\bappellant(?:\'s)? (?:case|stand|position|contention)\b',
                    r'\bon behalf of (?:the )?(?:petitioner|appellant)\b',
                ],
                'weak': [
                    r'\bpetitioner\b',
                    r'\bappellant\b',
                ]
            },
            
            'Arguments_Respondent': {
                'strong': [
                    r'\b(?:learned )?(?:counsel|advocate|senior advocate) (?:for|appearing for|representing) (?:the )?(?:respondent|defendant|accused)\b',
                    r'\brespondent(?:\'s)? (?:contends?|submits?|argues?|pleads?|claims?)\b',
                    r'\bdefendant(?:\'s)? (?:contends?|submits?|argues?|pleads?|claims?)\b',
                    r'\b(?:it is|it was) (?:contended|submitted|argued|urged|pleaded) (?:by|on behalf of) (?:the )?(?:respondent|defendant)\b',
                    r'\blearned counsel for (?:the )?(?:respondent|defendant) (?:has )?(?:contended|submitted|argued|urged)\b',
                ],
                'medium': [
                    r'\brespondent(?:\'s)? (?:case|stand|position|contention)\b',
                    r'\bdefendant(?:\'s)? (?:case|stand|position|contention)\b',
                    r'\bon behalf of (?:the )?(?:respondent|defendant)\b',
                ],
                'weak': [
                    r'\brespondent\b',
                    r'\bdefendant\b',
                ]
            },
            
            'Precedent_Analysis': {
                'strong': [
                    r'\bin\s+[A-Z][^v]{2,50}\s+v\.?\s+[A-Z][^,\(]{2,50}[,\s]*\([^\)]*\d{4}[^\)]*\)',
                    r'\bthe (?:Hon\'ble )?(?:Supreme Court|High Court) in [A-Z]',
                    r'\breference (?:is|may be) made to\b',
                    r'\bin the said (?:case|judgment|decision)\b',
                    r'\b(?:their Lordships?|the Court) (?:held|observed|ruled|noted|opined)\b',
                    r'\b(?:held|observed|ruled) in [A-Z]',
                    r'\breliance (?:is|was) placed on\b',
                    r'\b(?:following|relying upon|guided by) the (?:judgment|decision|ratio)\b',
                ],
                'medium': [
                    r'\b(?:AIR|SCC|SCR)\s+\d{4}\b',
                    r'\blaid down (?:in|by)\b',
                    r'\bprecedent\b',
                    r'\b(?:followed|distinguished|overruled)\b',
                    r'\bratio decidendi\b',
                ],
                'weak': [
                    r'\bcited\b',
                    r'\bcase law\b',
                ]
            },
            
            'Section_Analysis': {
                'strong': [
                    r'\bSection \d+[\w\(\)]* (?:of (?:the )?[A-Z][^\.,]{5,80}(?:Act|Code))\b',
                    r'\bArticle \d+[\w\(\)]* (?:of (?:the )?Constitution)\b',
                    r'\b(?:the|said|this) (?:section|article|provision) (?:reads|provides|states|stipulates|mandates)\b',
                    r'\b(?:bare|plain|literal|textual) reading of (?:Section|Article)\b',
                    r'\b(?:interpretation|scope|ambit|applicability) of (?:Section|Article)\b',
                    r'\blegislative (?:intent|history|scheme)\b',
                ],
                'medium': [
                    r'\b(?:Section|Sections|Article|Articles) \d+',
                    r'\bstatutory provision\b',
                    r'\bAct (?:provides|mandates|requires|stipulates)\b',
                    r'\b(?:proviso|explanation|sub-section|clause)\b',
                ],
                'weak': [
                    r'\bprovision of law\b',
                    r'\bunder the Act\b',
                ]
            },
            
            'Issues': {
                'strong': [
                    r'\b(?:question|issue|point)s? (?:for|of) (?:consideration|determination|decision)\b',
                    r'\b(?:legal|substantial) question of law\b',
                    r'\b(?:the|following) (?:question|issue)s? (?:arise|arises|fall|falls)\b',
                    r'\bwhether the (?:order|judgment|decision|action)\b',
                    r'\bpoint of law\b',
                    r'\bmoot question\b',
                ],
                'medium': [
                    r'\bquestion (?:raised|involved|posed)\b',
                    r'\bissue (?:raised|involved|posed)\b',
                    r'\b(?:narrow|limited|short) question\b',
                ],
                'weak': [
                    r'\bquestion\b',
                    r'\bissue\b',
                ]
            },
            
            'Court_Reasoning': {
                'strong': [
                    r'\b(?:we|the Court) (?:are|is) of the (?:view|opinion|considered view)\b',
                    r'\b(?:in our|in the Court\'s) (?:view|opinion|judgment|considered opinion)\b',
                    r'\b(?:we|the Court) (?:find|hold|note|observe) that\b',
                    r'\bit (?:is|appears|seems) (?:clear|evident|manifest|obvious) that\b',
                    r'\b(?:thus|therefore|hence|consequently|accordingly),? (?:we|the Court|it)\b',
                    r'\b(?:having )?(?:considered|examined|analyzed|perused) the (?:material|evidence|submissions|arguments)\b',
                    r'\b(?:taking|bearing) in (?:view|mind|consideration)\b',
                ],
                'medium': [
                    r'\b(?:we|the Court) (?:are|is) (?:satisfied|convinced|of the opinion)\b',
                    r'\bin the (?:light|backdrop|context) of\b',
                    r'\b(?:keeping in view|considering|taking into account)\b',
                    r'\bfor the (?:reasons|grounds) (?:stated|discussed|aforementioned)\b',
                ],
                'weak': [
                    r'\b(?:we|the Court) (?:observe|note)\b',
                    r'\bit is (?:seen|found|noted)\b',
                ]
            },
            
            'Conclusion': {
                'strong': [
                    r'\b(?:appeal|petition|writ petition|suit|application) (?:is|stands?) (?:allowed|dismissed|disposed of|partly allowed)\b',
                    r'\b(?:in the result|in view of the above|for the foregoing reasons)\b',
                    r'\b(?:we|the Court) (?:accordingly|therefore|thus) (?:allow|dismiss|dispose|set aside|quash|remand)\b',
                    r'\bfinal order\b',
                    r'\b(?:let|the) (?:decree|order) be drawn\b',
                    r'\b(?:no|with) (?:order as to )?costs?\b',
                ],
                'medium': [
                    r'\b(?:allowed|dismissed|disposed) accordingly\b',
                    r'\b(?:set aside|quashed|remanded|restored)\b',
                    r'\bimpugned (?:order|judgment) (?:is|stands)\b',
                ],
                'weak': [
                    r'\bconclusion\b',
                    r'\bfinally\b',
                ]
            }
        }
        
        # Weights for scoring
        self.weights = {'strong': 3.0, 'medium': 1.5, 'weak': 0.5}
    
    def classify_sentence(self, sentence: str, context_before: str = "", context_after: str = "") -> Tuple[str, float]:
        """
        Classify a single sentence into one of 8 categories.
        Uses weighted pattern matching with context awareness.
        """
        scores = {category: 0.0 for category in self.category_patterns.keys()}
        
        # Combine sentence with limited context
        full_text = f"{context_before[-200:]} {sentence} {context_after[:200]}"
        sentence_lower = sentence.lower()
        
        # Score each category
        for category, patterns in self.category_patterns.items():
            for strength, pattern_list in patterns.items():
                for pattern in pattern_list:
                    # Check in sentence first (higher weight)
                    if re.search(pattern, sentence, re.IGNORECASE):
                        scores[category] += self.weights[strength] * 1.5
                    # Check in context (lower weight)
                    elif re.search(pattern, full_text, re.IGNORECASE):
                        scores[category] += self.weights[strength] * 0.5
        
        # Positional heuristics (conclusions typically at end)
        # This would be enhanced with document position info in real implementation
        
        # Get top category
        if max(scores.values()) == 0:
            return 'Court_Reasoning', 0.1  # Default to reasoning if no patterns match
        
        top_category = max(scores.items(), key=lambda x: x[1])
        
        # Normalize confidence (0-1 scale)
        max_possible_score = 10.0  # Approximate max
        confidence = min(top_category[1] / max_possible_score, 1.0)
        
        return top_category[0], confidence


class SemanticLegalChunker:
    """
    Semantic-aware chunker that groups sentences by legal category.
    Creates chunks where each chunk belongs to a single semantic class.
    """
    
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1500,
        merge_threshold: int = 150,
        context_window: int = 3,
        confidence_threshold: float = 0.2,
    ):
        """
        Args:
            min_chunk_size: Minimum characters per chunk
            max_chunk_size: Maximum characters per chunk (split if exceeded)
            merge_threshold: Merge tiny consecutive chunks of same category
            context_window: Number of sentences for context in classification
            confidence_threshold: Minimum confidence to use classification
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.merge_threshold = merge_threshold
        self.context_window = context_window
        self.confidence_threshold = confidence_threshold
        
        self.classifier = LegalSemanticClassifier()
        
        logger.info(
            f"[INIT] SemanticLegalChunker(min={min_chunk_size}, max={max_chunk_size}, "
            f"merge_threshold={merge_threshold})"
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling legal text peculiarities."""
        # Protect legal abbreviations
        protected = text
        protected = re.sub(r'\bv\.\s', 'v_PROTECT_ ', protected)
        protected = re.sub(r'\b(Hon\'ble|Mr|Mrs|Ms|Dr|S|Art|Sec|Para|Cl)\.\s', r'\1_PROTECT_ ', protected)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', protected)
        
        # Restore protected text
        sentences = [s.replace('_PROTECT_', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_all_sentences(self, sentences: List[str]) -> List[SemanticUnit]:
        """Classify all sentences with context."""
        classified_units = []
        
        for i, sentence in enumerate(sentences):
            # Get context
            context_before = ' '.join(sentences[max(0, i - self.context_window):i])
            context_after = ' '.join(sentences[i + 1:min(len(sentences), i + self.context_window + 1)])
            
            # Classify
            category, confidence = self.classifier.classify_sentence(
                sentence, context_before, context_after
            )
            
            classified_units.append(SemanticUnit(
                text=sentence,
                index=i,
                category=category,
                confidence=confidence
            ))
        
        return classified_units
    
    def _smooth_classifications(self, units: List[SemanticUnit]) -> List[SemanticUnit]:
        """
        Smooth classifications using sliding window majority voting.
        Reduces noise from misclassified sentences.
        """
        if len(units) < 3:
            return units
        
        smoothed = []
        window_size = 5
        
        for i, unit in enumerate(units):
            # Get window
            start = max(0, i - window_size // 2)
            end = min(len(units), i + window_size // 2 + 1)
            window = units[start:end]
            
            # If current confidence is high, keep it
            if unit.confidence > 0.6:
                smoothed.append(unit)
                continue
            
            # Otherwise, use majority vote in window
            categories = [u.category for u in window]
            majority = Counter(categories).most_common(1)[0][0]
            
            smoothed.append(SemanticUnit(
                text=unit.text,
                index=unit.index,
                category=majority,
                confidence=unit.confidence * 0.8  # Slightly reduce confidence
            ))
        
        return smoothed
    
    def _group_into_chunks(self, units: List[SemanticUnit]) -> List[List[SemanticUnit]]:
        """
        Group consecutive sentences of same category into chunks.
        Respects size constraints.
        """
        if not units:
            return []
        
        chunks = []
        current_chunk = [units[0]]
        current_size = len(units[0].text)
        
        for unit in units[1:]:
            same_category = unit.category == current_chunk[0].category
            would_fit = current_size + len(unit.text) <= self.max_chunk_size
            
            if same_category and would_fit:
                # Add to current chunk
                current_chunk.append(unit)
                current_size += len(unit.text)
            else:
                # Save current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = [unit]
                current_size = len(unit.text)
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _merge_tiny_chunks(self, chunks: List[List[SemanticUnit]]) -> List[List[SemanticUnit]]:
        """
        Merge very small consecutive chunks of same category.
        """
        if not chunks:
            return []
        
        merged = [chunks[0]]
        
        for chunk in chunks[1:]:
            prev_chunk = merged[-1]
            chunk_text = ' '.join(u.text for u in chunk)
            prev_text = ' '.join(u.text for u in prev_chunk)
            
            same_category = chunk[0].category == prev_chunk[0].category
            both_small = len(chunk_text) < self.merge_threshold and len(prev_text) < self.merge_threshold
            prev_small = len(prev_text) < self.merge_threshold
            
            if same_category and (both_small or prev_small):
                # Merge with previous
                merged[-1].extend(chunk)
            else:
                merged.append(chunk)
        
        return merged
    
    def _split_oversized_chunks(self, chunks: List[List[SemanticUnit]]) -> List[List[SemanticUnit]]:
        """
        Split chunks that exceed max_chunk_size while maintaining category.
        """
        split_chunks = []
        
        for chunk in chunks:
            chunk_text = ' '.join(u.text for u in chunk)
            
            if len(chunk_text) <= self.max_chunk_size:
                split_chunks.append(chunk)
            else:
                # Split into smaller sub-chunks
                sub_chunk = []
                sub_size = 0
                
                for unit in chunk:
                    if sub_size + len(unit.text) <= self.max_chunk_size:
                        sub_chunk.append(unit)
                        sub_size += len(unit.text)
                    else:
                        if sub_chunk:
                            split_chunks.append(sub_chunk)
                        sub_chunk = [unit]
                        sub_size = len(unit.text)
                
                if sub_chunk:
                    split_chunks.append(sub_chunk)
        
        return split_chunks
    
    def chunk_document(self, text: str) -> List[Dict]:
        """
        Main method: Create semantically coherent chunks with category labels.
        """
        if not isinstance(text, str) or not text.strip():
            logger.warning("[CHUNKER] Empty or invalid text.")
            return []
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        logger.info(f"[CHUNKER] Split into {len(sentences)} sentences.")
        
        if not sentences:
            return []
        
        # Step 2: Classify each sentence
        classified_units = self._classify_all_sentences(sentences)
        logger.info(f"[CHUNKER] Classified {len(classified_units)} units.")
        
        # Step 3: Smooth classifications
        smoothed_units = self._smooth_classifications(classified_units)
        
        # Log category distribution
        category_dist = Counter(u.category for u in smoothed_units)
        logger.info(f"[CHUNKER] Category distribution: {dict(category_dist)}")
        
        # Step 4: Group into chunks
        chunks = self._group_into_chunks(smoothed_units)
        logger.info(f"[CHUNKER] Grouped into {len(chunks)} initial chunks.")
        
        # Step 5: Merge tiny chunks
        chunks = self._merge_tiny_chunks(chunks)
        logger.info(f"[CHUNKER] After merging: {len(chunks)} chunks.")
        
        # Step 6: Split oversized chunks
        chunks = self._split_oversized_chunks(chunks)
        logger.info(f"[CHUNKER] After splitting: {len(chunks)} final chunks.")
        
        # Step 7: Build final output with metadata
        final_chunks = []
        for i, chunk_units in enumerate(chunks):
            chunk_text = ' '.join(u.text for u in chunk_units)
            category = chunk_units[0].category
            avg_confidence = sum(u.confidence for u in chunk_units) / len(chunk_units)
            
            chunk_data = {
                "chunk_id": str(i),
                "text": chunk_text,
                "type": category,  # One of the 8 categories
                "char_count": len(chunk_text),
                "sentence_count": len(chunk_units),
                "confidence": round(avg_confidence, 3),
                "sentence_indices": [u.index for u in chunk_units],
                
                # Additional metadata
                "has_citations": bool(re.search(r'\(\d{4}\)\s+\d+\s+(?:SCC|AIR)', chunk_text)),
                "has_statutes": bool(re.search(r'\b(?:Section|Article)\s+\d+', chunk_text)),
                "has_parties": bool(re.search(r'\b(?:petitioner|respondent|appellant|defendant)', chunk_text, re.IGNORECASE)),
            }
            
            final_chunks.append(chunk_data)
        
        # Log final statistics
        type_dist = Counter(c['type'] for c in final_chunks)
        avg_size = sum(c['char_count'] for c in final_chunks) / len(final_chunks)
        logger.info(
            f"[CHUNKER] ✅ Generated {len(final_chunks)} semantic chunks. "
            f"Avg size: {avg_size:.0f} chars. Type dist: {dict(type_dist)}"
        )
        
        return final_chunks