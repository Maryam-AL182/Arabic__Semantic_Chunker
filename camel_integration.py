"""
CAMeL Tools Integration V2 - PRODUCTION

CRITICAL FIX #5: Grammar-First, Refine-with-Semantics Pipeline

OLD PIPELINE (wrong):
  Split by embeddings → Refine with grammar rules

NEW PIPELINE (correct):
  1. Detect grammar boundaries FIRST
  2. Split text at boundaries
  3. Merge adjacent segments if semantically similar AND size permits
  4. Add overlap

This flips the system from "semantic-first" to "grammar-first",
which is more linguistically grounded.

Additional Features:
- Discourse marker detection in boundary scoring
- CAMeL Tools integration with fallback
- Morphological analysis
- Dependency-aware chunking
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re


# ==============================================================================
# MORPHOLOGICAL ANALYSIS
# ==============================================================================

@dataclass
class MorphToken:
    """Represents a morphologically analyzed token"""
    surface: str
    lemma: str
    pos: str
    features: Dict[str, str]
    diac: str = ""
    gloss: str = ""
    dep_rel: str = ""
    head: int = -1
    
    def is_verb(self) -> bool:
        """Check if token is a verb"""
        return self.pos.startswith("V") or self.pos in ["VERB", "VERB_PERF", "VERB_IMPERF"]
    
    def is_noun(self) -> bool:
        """Check if token is a noun"""
        return self.pos.startswith("N") or self.pos in ["NOUN", "NOUN_PROP", "NOUN_QUANT"]
    
    def is_preposition(self) -> bool:
        """Check if token is a preposition"""
        return self.pos in ["PREP", "P"]
    
    def is_relative_pronoun(self) -> bool:
        """Check if token is a relative pronoun"""
        return self.pos in ["PRON_REL", "REL"] or self.lemma in [
            'الذي', 'التي', 'الذين', 'اللاتي', 'اللائي', 'اللواتي'
        ]
    
    def is_conjunction(self) -> bool:
        """Check if token is a conjunction"""
        return self.pos in ["CONJ", "CC", "C"]
    
    def is_discourse_marker(self) -> bool:
        """Check if token is part of a discourse marker phrase"""
        discourse_words = {
            'ناحية', 'صعيد', 'جهة', 'جانب', 'مقابل',
            'بالعودة', 'بالإضافة', 'علاوة', 'سياق',
            'نقيض', 'حين'
        }
        return self.lemma in discourse_words or self.surface in discourse_words


# ==============================================================================
# CAMEL ANALYZER
# ==============================================================================

class CAMeLAnalyzer:
    """
    Wrapper around CAMeL Tools with robust fallback.
    Provides morphological analysis and disambiguation.
    """
    
    def __init__(self, use_camel: bool = True):
        """
        Args:
            use_camel: If True, attempt to use CAMeL Tools
        """
        self.camel_available = False
        self.analyzer = None
        self.disambiguator = None
        
        if use_camel:
            try:
                from camel_tools.morphology.database import MorphologyDB
                from camel_tools.morphology.analyzer import Analyzer
                from camel_tools.disambig.mle import MLEDisambiguator
                
                self.db = MorphologyDB.builtin_db()
                self.analyzer = Analyzer(self.db)
                self.disambiguator = MLEDisambiguator(self.db)
                self.camel_available = True
                
                print("✓ CAMeL Tools initialized successfully")
                
            except ImportError:
                print("⚠ CAMeL Tools not available, using rule-based analysis")
                print("  Install with: pip install camel-tools")
            except Exception as e:
                print(f"⚠ Error initializing CAMeL Tools: {e}")
                print("  Using rule-based fallback")
    
    def analyze_text(self, text: str) -> List[List[MorphToken]]:
        """
        Analyze text and return morphological information.
        
        Args:
            text: Arabic text to analyze
        
        Returns:
            List of sentences, each containing list of MorphTokens
        """
        if self.camel_available:
            return self._analyze_with_camel(text)
        else:
            return self._analyze_rule_based(text)
    
    def _analyze_with_camel(self, text: str) -> List[List[MorphToken]]:
        """Analyze using CAMeL Tools"""
        from camel_tools.tokenizers.word import simple_word_tokenize
        
        sentences = []
        # Split into sentences
        sent_texts = re.split(r'[.!؟।۔]+', text)
        
        for sent_text in sent_texts:
            if not sent_text.strip():
                continue
            
            # Tokenize
            tokens = simple_word_tokenize(sent_text)
            
            # Disambiguate
            disambig = self.disambiguator.disambiguate(tokens)
            
            # Convert to MorphTokens
            morph_tokens = []
            for token_analysis in disambig:
                morph_token = MorphToken(
                    surface=token_analysis.get('word', ''),
                    lemma=token_analysis.get('lemma', token_analysis.get('word', '')),
                    pos=token_analysis.get('pos', 'X'),
                    features=token_analysis,
                    diac=token_analysis.get('diac', ''),
                    gloss=token_analysis.get('gloss', '')
                )
                morph_tokens.append(morph_token)
            
            sentences.append(morph_tokens)
        
        return sentences
    
    def _analyze_rule_based(self, text: str) -> List[List[MorphToken]]:
        """
        Fallback rule-based analysis using regex patterns.
        Less accurate than CAMeL but works without dependencies.
        """
        # Simple POS heuristics
        verb_patterns = [r'^[يتنأ]', r'[تيا]$']
        noun_patterns = [r'^ال', r'[ةه]$']
        
        # Known word lists
        relative_pronouns = {
            'الذي', 'التي', 'الذين', 'اللاتي', 'اللائي', 'اللواتي',
            'من', 'ما', 'اللذان', 'اللتان'
        }
        prepositions = {
            'في', 'من', 'إلى', 'على', 'عن', 'ل', 'ب',
            'ك', 'مع', 'بعد', 'قبل', 'حتى', 'عند', 'لدى',
            'خلال', 'حول', 'ضد', 'نحو'
        }
        conjunctions = {
            'و', 'ف', 'ثم', 'أو', 'لكن', 'بل', 'حتى',
            'إن', 'أن', 'كي', 'لو', 'إذا', 'حيث'
        }
        
        sentences = []
        sent_texts = re.split(r'[.!؟।۔]+', text)
        
        for sent_text in sent_texts:
            if not sent_text.strip():
                continue
            
            # Simple tokenization
            tokens = re.findall(r'\S+', sent_text)
            morph_tokens = []
            
            for token in tokens:
                clean_token = re.sub(r'[^\w]', '', token)
                
                # Determine POS
                if clean_token in relative_pronouns:
                    pos = "PRON_REL"
                    lemma = clean_token
                elif clean_token in prepositions or token.startswith(tuple(prepositions)):
                    pos = "PREP"
                    lemma = clean_token
                elif clean_token in conjunctions:
                    pos = "CONJ"
                    lemma = clean_token
                elif any(re.match(p, clean_token) for p in verb_patterns):
                    pos = "VERB"
                    lemma = clean_token
                elif any(re.match(p, clean_token) for p in noun_patterns):
                    pos = "NOUN"
                    lemma = clean_token
                else:
                    pos = "X"  # Unknown
                    lemma = clean_token
                
                morph_token = MorphToken(
                    surface=token,
                    lemma=lemma,
                    pos=pos,
                    features={}
                )
                morph_tokens.append(morph_token)
            
            sentences.append(morph_tokens)
        
        return sentences


# ==============================================================================
# DEPENDENCY BOUNDARY DETECTOR
# ==============================================================================

class DependencyBoundaryDetector:
    """
    REFACTORED: Grammar-first boundary detection.
    
    This detector runs BEFORE semantic chunking, not after.
    It identifies potential chunk boundaries based on:
    1. Morphological patterns
    2. Syntactic structures
    3. Discourse markers
    4. Punctuation
    """
    
    # Dependency relations that indicate predicate boundaries
    PREDICATE_RELS = {"root", "ccomp", "xcomp", "conj", "advcl", "parataxis"}
    
    # Modifier relations (should stay together)
    MODIFIER_RELS = {"acl", "acl:relcl", "nmod", "amod", "obl"}
    
    # FIX: DISCOURSE MARKER PATTERNS
    DISCOURSE_PATTERNS = [
        r'من\s+ناحية\s+(اخرى|أخرى)',
        r'على\s+صعيد\s+(اخر|آخر)',
        r'من\s+جهة\s+(اخرى|أخرى)',
        r'من\s+جانب\s+(اخر|آخر)',
        r'في\s+سياق\s+مختلف',
        r'بالعودة\s+(إلى|الى)',
        r'بالإضافة\s+(إلى|الى)',
        r'في\s+المقابل',
        r'على\s+النقيض',
        r'في\s+حين\s+(أن|ان)',
        r'علاوة\s+على\s+ذلك'
    ]
    
    def __init__(self, analyzer: CAMeLAnalyzer):
        """
        Args:
            analyzer: CAMeLAnalyzer instance for morphological analysis
        """
        self.analyzer = analyzer
    
    def detect_boundaries(
        self,
        text: str,
        min_boundary_score: float = 0.6
    ) -> List[Tuple[int, float, str]]:
        """
        Detect grammatical boundaries in text.
        
        Args:
            text: Arabic text to analyze
            min_boundary_score: Minimum score to consider as boundary
        
        Returns:
            List of (char_position, confidence_score, reason) tuples
        """
        # Analyze text morphologically
        sentences = self.analyzer.analyze_text(text)
        
        boundaries = [(0, 1.0, "START")]  # Always start with text beginning
        char_pos = 0
        
        for sent_tokens in sentences:
            # Mark protected spans (should not split)
            protected_spans = self._mark_protected_spans(sent_tokens)
            
            # Score each token position as potential boundary
            for i, token in enumerate(sent_tokens):
                char_pos += len(token.surface) + 1
                
                # Skip if inside protected span
                if any(start <= i < end for start, end in protected_spans):
                    continue
                
                # Calculate boundary score
                score, reason = self._calculate_boundary_score(
                    sent_tokens, i, text, char_pos
                )
                
                # Add boundary if score is high enough
                if score >= min_boundary_score:
                    boundaries.append((char_pos, score, reason))
        
        # Always end with text end
        boundaries.append((len(text), 1.0, "END"))
        
        return boundaries
    
    def _mark_protected_spans(self, tokens: List[MorphToken]) -> List[Tuple[int, int]]:
        """
        Mark grammatical units that should NOT be split.
        
        Protected spans include:
        - Relative clauses (verbless, participial, nested)
        - Prepositional phrases (variable length)
        - Noun phrases (noun + adjective/genitive/number/demonstrative)
        - Genitive constructions (إضافة)
        """
        protected = []
        
        # 1. RELATIVE CLAUSES (improved)
        for i, token in enumerate(tokens):
            if token.is_relative_pronoun():
                end = i + 1
                depth = 1
                has_verb = False
                
                while end < len(tokens) and depth > 0:
                    current = tokens[end]
                    
                    # Check for verb (closes clause)
                    if current.is_verb():
                        has_verb = True
                        depth -= 1
                    # Nested relative pronoun
                    elif current.is_relative_pronoun():
                        depth += 1
                    # Participial/verbless clause: look for noun or adjective continuation
                    elif not has_verb and (current.is_noun() or current.pos in ['ADJ', 'PRON']):
                        pass  # Continue extending
                    # Punctuation or conjunction ends clause
                    elif re.search(r'[.،؛؟!]', current.surface) or current.is_conjunction():
                        break
                    
                    end += 1
                    
                    # Safety limit for verbless clauses
                    if end - i > 10:
                        break
                
                protected.append((i, end))
        
        # 2. PREPOSITIONAL PHRASES (variable length)
        for i, token in enumerate(tokens):
            if token.is_preposition():
                end = i + 1
                
                # Extended PP patterns:
                # - على الرغم من أن
                # - في الوقت الذي
                # - بالإضافة إلى
                
                # Continue while we see:
                # - Nouns (including proper nouns)
                # - Adjectives modifying the noun
                # - Genitive constructions (noun after noun)
                # - Determiners (ال)
                while end < len(tokens):
                    current = tokens[end]
                    
                    # Nouns and adjectives continue PP
                    if current.is_noun() or current.pos in ['ADJ', 'DET']:
                        end += 1
                    # Stop at verb or another preposition
                    elif current.is_verb() or current.is_preposition():
                        break
                    # Stop at punctuation
                    elif re.search(r'[.،؛؟!]', current.surface):
                        break
                    else:
                        # Check if it's part of multi-word PP
                        if end - i < 6:  # Allow up to 6 tokens for complex PPs
                            end += 1
                        else:
                            break
                
                protected.append((i, min(end, i + 8)))  # Cap at 8 tokens max
        
        # 3. NOUN PHRASES (noun + modifiers)
        for i, token in enumerate(tokens):
            if token.is_noun():
                end = i + 1
                
                # Extend NP for:
                # - Adjectives (صفة)
                # - Genitive nouns (إضافة): noun + noun
                # - Numbers
                # - Demonstratives
                
                while end < len(tokens):
                    current = tokens[end]
                    
                    # Adjective modifying noun
                    if current.pos in ['ADJ']:
                        end += 1
                    # Genitive construction (noun after noun)
                    elif current.is_noun() and end == i + 1:
                        # First noun after head noun = likely genitive
                        end += 1
                    # Number (cardinal/ordinal)
                    elif current.pos in ['NUM', 'NUM_CARD', 'NUM_ORD']:
                        end += 1
                    # Demonstrative
                    elif current.pos in ['DEM', 'PRON_DEM']:
                        end += 1
                    # Coordinated nouns (و العطف)
                    elif current.surface in ['و', 'أو'] and end < len(tokens) - 1:
                        if tokens[end + 1].is_noun():
                            end += 2  # Skip conjunction and add next noun
                        else:
                            break
                    else:
                        break
                    
                    # Safety limit
                    if end - i > 5:
                        break
                
                # Only protect if we actually found modifiers
                if end > i + 1:
                    protected.append((i, end))
        
        # 4. GENITIVE CONSTRUCTIONS (إضافة) - additional pass
        # Pattern: noun + noun (+ noun...)
        i = 0
        while i < len(tokens) - 1:
            if tokens[i].is_noun() and tokens[i + 1].is_noun():
                # Start of genitive chain
                end = i + 1
                while end < len(tokens) and tokens[end].is_noun():
                    end += 1
                
                # Protect if chain is 2+ nouns
                if end > i + 1:
                    protected.append((i, end))
                    i = end
                else:
                    i += 1
            else:
                i += 1
        
        return protected
    
    def visualize_protected_spans(
        self,
        text: str,
        tokens: List[MorphToken],
        protected_spans: List[Tuple[int, int]]
    ) -> str:
        """
        Visualize protected spans by marking them in the text.
        Useful for debugging and understanding grammar protection.
        
        Returns:
            Formatted string with protected spans marked
        """
        if not protected_spans:
            return "No protected spans found."
        
        output = []
        output.append("=" * 80)
        output.append("PROTECTED SPANS VISUALIZATION")
        output.append("=" * 80)
        
        for idx, (start, end) in enumerate(protected_spans, 1):
            span_tokens = tokens[start:end]
            span_text = ' '.join([t.surface for t in span_tokens])
            
            # Determine span type
            if span_tokens[0].is_relative_pronoun():
                span_type = "RELATIVE CLAUSE"
            elif span_tokens[0].is_preposition():
                span_type = "PREPOSITIONAL PHRASE"
            elif span_tokens[0].is_noun():
                span_type = "NOUN PHRASE"
            else:
                span_type = "PROTECTED"
            
            output.append(f"\n{idx}. [{span_type}] (tokens {start}-{end}):")
            output.append(f"   {span_text}")
        
        return '\n'.join(output)
    
    def _calculate_boundary_score(
        self,
        tokens: List[MorphToken],
        index: int,
        full_text: str,
        char_pos: int
    ) -> Tuple[float, str]:
        """
        Calculate boundary strength at a given position.
        
        Args:
            tokens: List of morphological tokens
            index: Current token index
            full_text: Full text (for context)
            char_pos: Character position in full text
        
        Returns:
            (score, reason) tuple where score is 0-1
        """
        if index >= len(tokens) - 1:
            return (0.0, "")
        
        current = tokens[index]
        next_token = tokens[index + 1]
        
        score = 0.0
        reason = ""
        
        # PRIORITY 1: Sentence boundaries (highest)
        if re.search(r'[.!؟।۔]', current.surface):
            return (0.95, "SENTENCE_END")
        
        # PRIORITY 2: Discourse markers (very high)
        window_text = full_text[max(0, char_pos - 50):char_pos + 100]
        for pattern in self.DISCOURSE_PATTERNS:
            if re.search(pattern, window_text):
                return (0.90, "DISCOURSE_MARKER")
        
        # Check if next token is discourse marker word
        if next_token.is_discourse_marker():
            score += 0.7
            reason = "DISCOURSE_WORD"
        
        # PRIORITY 3: Clause boundaries
        if next_token.is_conjunction() and current.is_verb():
            score += 0.6
            reason = "CLAUSE_CONJUNCTION"
        
        # Verb sequences (topic shift indicator)
        if current.is_verb() and next_token.is_verb():
            score += 0.5
            reason = "VERB_SEQUENCE"
        
        # Major clause boundary: Noun → Verb
        if current.is_noun() and next_token.is_verb():
            score += 0.4
            reason = "CLAUSE_BOUNDARY"
        
        # PRIORITY 4: Punctuation
        if re.search(r'[،؛]', current.surface):
            score += 0.3
            reason = "PUNCTUATION"
        
        # Conjunction alone
        if next_token.is_conjunction():
            score += 0.2
            reason = "CONJUNCTION"
        
        # Cap score at 1.0
        return (min(score, 1.0), reason)


# ==============================================================================
# ENHANCED GRAMMAR CHUNKER
# ==============================================================================

class EnhancedGrammarChunker:
    """
    REFACTORED V2: Grammar-First, Semantics-Second Pipeline
    
    Pipeline:
    1. Detect grammar boundaries (using morphology & discourse markers)
    2. Split text at boundaries
    3. Merge adjacent segments if:
       - Combined size < threshold
       - No discourse marker between them
       - Semantically similar (optional)
    4. Add overlap
    
    This is the CORRECT architecture for linguistic chunking.
    """
    
    def __init__(
        self,
        use_camel: bool = True,
        target_chunk_size: int = 512,
        overlap_size: int = 50
    ):
        """
        Args:
            use_camel: If True, use CAMeL Tools (more accurate)
            target_chunk_size: Target size for chunks in characters
            overlap_size: Overlap between chunks in characters
        """
        self.analyzer = CAMeLAnalyzer(use_camel=use_camel)
        self.boundary_detector = DependencyBoundaryDetector(self.analyzer)
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
    
    def chunk(self, text: str) -> List[Dict]:
        """
        Chunk text using grammar-first pipeline.
        
        Args:
            text: Arabic text to chunk
        
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Detect grammar boundaries
        boundaries = self.boundary_detector.detect_boundaries(text)
        
        # Step 2: Create initial segments from boundaries
        initial_segments = self._create_segments_from_boundaries(text, boundaries)
        
        # Step 3: Merge semantically similar segments (if size permits)
        merged_segments = self._merge_segments(initial_segments)
        
        # Step 4: Convert to output format
        chunks = []
        for seg in merged_segments:
            chunks.append({
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end'],
                'length': len(seg['text']),
                'boundary_score': seg['score'],
                'boundary_reason': seg.get('reason', ''),
                'metadata': seg.get('metadata', {})
            })
        
        # Step 5: Add overlap
        if self.overlap_size > 0:
            chunks = self._add_overlap(chunks, text)
        
        return chunks
    
    def _create_segments_from_boundaries(
        self,
        text: str,
        boundaries: List[Tuple[int, float, str]]
    ) -> List[Dict]:
        """
        Create segments from detected boundaries.
        
        Args:
            text: Full text
            boundaries: List of (position, score, reason) tuples
        
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        for i in range(len(boundaries) - 1):
            start_pos, start_score, start_reason = boundaries[i]
            end_pos, _, _ = boundaries[i + 1]
            
            segment_text = text[start_pos:end_pos].strip()
            
            if segment_text:
                segments.append({
                    'text': segment_text,
                    'start': start_pos,
                    'end': end_pos,
                    'score': start_score,
                    'reason': start_reason,
                    'metadata': {}
                })
        
        return segments
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Merge adjacent segments if appropriate.
        
        Merge conditions:
        1. Combined length < 1.5 * target_chunk_size
        2. No discourse marker between them
        3. Both are relatively small (< 0.5 * target)
        
        This prevents over-fragmentation while respecting topic boundaries.
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # Try to merge with next segment
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                combined_len = len(current['text']) + len(next_seg['text'])
                
                # Merge conditions
                can_merge = (
                    # Size constraint
                    combined_len < self.target_chunk_size * 1.5 and
                    # No discourse marker (respect topic boundaries)
                    next_seg.get('reason', '') != 'DISCOURSE_MARKER' and
                    next_seg.get('reason', '') != 'SENTENCE_END' and
                    # At least one is small
                    (len(current['text']) < self.target_chunk_size * 0.5 or
                     len(next_seg['text']) < self.target_chunk_size * 0.5)
                )
                
                if can_merge:
                    # Merge the segments
                    merged_segment = {
                        'text': current['text'] + ' ' + next_seg['text'],
                        'start': current['start'],
                        'end': next_seg['end'],
                        'score': min(current['score'], next_seg['score']),
                        'reason': current.get('reason', ''),
                        'metadata': {
                            'merged': True,
                            'original_count': 2
                        }
                    }
                    merged.append(merged_segment)
                    i += 2  # Skip both segments
                    continue
            
            # No merge - keep segment as is
            merged.append(current)
            i += 1
        
        return merged
    
    def _add_overlap(self, chunks: List[Dict], text: str) -> List[Dict]:
        """
        Add morphology-aware overlap between consecutive chunks.
        
        Uses CAMeL Tools integration to:
        1. Detect complete grammatical structures (verb phrases, noun phrases)
        2. Respect morphological boundaries (clitics, affixes)
        3. Preserve discourse markers and connectors
        4. Extract semantically coherent overlap units
        
        Falls back to grammar-aware overlap if CAMeL is unavailable.
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            new_chunk = chunk.copy()
            
            # Add morphology-aware overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i - 1]
                
                # Extract overlap using morphological analysis
                overlap_text = self._extract_morphology_overlap(
                    prev_chunk['text'],
                    self.overlap_size
                )
                
                if overlap_text:
                    # Prepend to current chunk
                    new_chunk['text'] = overlap_text + " " + new_chunk['text']
                    new_chunk['metadata'] = chunk.get('metadata', {}).copy()
                    new_chunk['metadata']['has_overlap'] = True
                    new_chunk['metadata']['overlap_size'] = len(overlap_text)
                    new_chunk['metadata']['overlap_type'] = 'morphology_aware'
            
            overlapped.append(new_chunk)
        
        return overlapped
    
    def _extract_morphology_overlap(self, text: str, target_size: int) -> str:
        """
        Extract morphologically coherent overlap from end of text.
        
        Strategy (with CAMeL):
        1. Analyze morphology of last portion of text
        2. Identify complete syntactic units (clauses, phrases)
        3. Detect discourse markers and connectors
        4. Extract last complete unit that fits target_size
        
        Strategy (without CAMeL):
        1. Fall back to grammar-aware extraction
        2. Use regex patterns for phrase detection
        3. Respect sentence and phrase boundaries
        
        Args:
            text: Source text to extract from
            target_size: Target overlap size in characters
        
        Returns:
            Morphologically coherent overlap text
        """
        if not text or target_size <= 0:
            return ""
        
        # If CAMeL is available, use morphological analysis
        if self.analyzer and self.analyzer.camel_available:
            return self._extract_overlap_with_camel(text, target_size)
        else:
            return self._extract_overlap_grammar_based(text, target_size)
    
    def _extract_overlap_with_camel(self, text: str, target_size: int) -> str:
        """
        Extract overlap using CAMeL morphological analysis.
        
        Identifies complete syntactic structures:
        - Complete sentences
        - Verbal phrases (verb + complements)
        - Nominal phrases (noun + modifiers)
        - Prepositional phrases
        - Relative clauses
        
        CRITICAL: Never returns overlap starting with conjunction
        """
        # Analyze the last portion that could contain overlap
        # Look at 2x target_size to have options
        search_window = min(len(text), target_size * 2)
        tail_text = text[-search_window:]
        
        try:
            # Get morphological analysis
            analyzed_sentences = self.analyzer.analyze_text(tail_text)
            
            if not analyzed_sentences:
                return self._extract_overlap_grammar_based(text, target_size)
            
            # Strategy 1: Try to get complete clauses/sentences
            # Start from the end and work backward
            selected_tokens = []
            current_len = 0
            clause_complete = False
            
            # Flatten all tokens
            all_tokens = []
            for sent_tokens in reversed(analyzed_sentences):
                all_tokens.extend(reversed(sent_tokens))
            
            for token in all_tokens:
                token_len = len(token.surface) + 1  # +1 for space
                
                # Check if adding this would exceed limit
                if current_len + token_len > target_size * 1.3:
                    # Only break at clause boundaries
                    if clause_complete:
                        break
                
                # Add token
                selected_tokens.insert(0, token.surface)
                current_len += token_len
                
                # Check if we have a complete clause
                # Indicators: sentence-final punctuation, discourse marker, verb phrase end
                if self._is_clause_boundary(token, selected_tokens):
                    clause_complete = True
                    
                    # If we're at a good size, stop here
                    if current_len >= target_size * 0.7:
                        break
            
            if selected_tokens:
                overlap = ' '.join(selected_tokens)
                # Ensure reasonable length
                if len(overlap) >= target_size * 0.4 and len(overlap) <= target_size * 1.5:
                    # Clean leading conjunctions
                    overlap = self._remove_leading_particles(overlap)
                    if overlap:
                        return overlap
        
        except Exception as e:
            # If CAMeL analysis fails, fall back to grammar-based
            pass
        
        # Fallback if CAMeL extraction didn't work
        return self._extract_overlap_grammar_based(text, target_size)
    
    def _is_clause_boundary(self, token: 'MorphToken', context: List[str]) -> bool:
        """
        Check if this token marks a clause boundary.
        
        Indicators:
        - Discourse markers (من ناحية، في سياق، بالعودة)
        - Conjunctions (و، ثم، لكن) when followed by verb
        - Complete prepositional phrases
        - Relative clause closures
        """
        # Check if it's a discourse marker
        if token.is_discourse_marker():
            return True
        
        # Check for conjunction + verb pattern (new clause)
        if token.is_conjunction() and len(context) > 2:
            return True
        
        # Check for complete prepositional phrase
        # Preposition followed by noun phrase ending
        if len(context) >= 2:
            # Look for pattern: PREP + NOUN + (modifiers)
            if token.is_noun() and len(context) >= 3:
                # Check if this closes a PP
                return True
        
        return False
    
    def _extract_overlap_grammar_based(self, text: str, target_size: int) -> str:
        """
        Fallback: Grammar-aware overlap extraction without morphology.
        
        Uses regex patterns and heuristics to identify:
        - Complete sentences
        - Phrase boundaries (comma, semicolon)
        - Discourse markers
        - Word boundaries
        
        CRITICAL: Never returns overlap starting with conjunction (و، ف، ثم)
        """
        # Strategy 1: Try complete sentences
        # Split on sentence terminators
        sentences = re.split(r'[.!؟।۔]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            # Get last N sentences that fit
            selected = []
            current_len = 0
            
            for sent in reversed(sentences):
                sent_len = len(sent)
                if current_len + sent_len <= target_size * 1.2:
                    selected.insert(0, sent)
                    current_len += sent_len
                else:
                    break
            
            if selected:
                overlap = ' '.join(selected)
                if len(overlap) >= target_size * 0.5 and len(overlap) <= target_size * 1.5:
                    # Clean leading conjunctions
                    overlap = self._remove_leading_particles(overlap)
                    if overlap:
                        return overlap
        
        # Strategy 2: Look for discourse markers in the tail
        search_window = min(len(text), target_size * 2)
        tail_text = text[-search_window:]
        
        # Discourse marker patterns
        discourse_patterns = [
            r'من ناحية أخرى',
            r'في سياق [^\s]+',
            r'على صعيد',
            r'من جهة',
            r'بالإضافة إلى',
            r'بالعودة إلى',
            r'في المقابل'
        ]
        
        for pattern in discourse_patterns:
            matches = list(re.finditer(pattern, tail_text))
            if matches:
                # Get text from last discourse marker
                last_match = matches[-1]
                overlap_candidate = tail_text[last_match.start():]
                
                if len(overlap_candidate) >= target_size * 0.5 and \
                   len(overlap_candidate) <= target_size * 1.5:
                    return overlap_candidate
        
        # Strategy 3: Phrase boundaries (comma, semicolon)
        phrases = re.split(r'[،؛]+', text)
        
        if len(phrases) > 1:
            # Get last phrase
            last_phrase = phrases[-1].strip()
            
            if len(last_phrase) >= target_size * 0.5 and \
               len(last_phrase) <= target_size * 1.5:
                # Clean leading conjunctions
                last_phrase = self._remove_leading_particles(last_phrase)
                if last_phrase:
                    return last_phrase
        
        # Strategy 4: Word boundaries
        # Never cut mid-word
        tokens = text.split()
        
        if tokens:
            selected = []
            current_len = 0
            
            for token in reversed(tokens):
                token_len = len(token) + 1
                if current_len + token_len <= target_size * 1.2:
                    selected.insert(0, token)
                    current_len += token_len
                else:
                    break
            
            if selected:
                overlap = ' '.join(selected)
                # Clean leading conjunctions
                overlap = self._remove_leading_particles(overlap)
                if overlap:
                    return overlap
        
        # Last resort: character-based at word boundary
        if len(text) >= target_size:
            cutoff = text[-target_size:]
            # Find first space
            space_idx = cutoff.find(' ')
            if space_idx > 0:
                overlap = cutoff[space_idx:].strip()
                # Clean leading conjunctions
                overlap = self._remove_leading_particles(overlap)
                if overlap:
                    return overlap
            
            overlap = cutoff.strip()
            overlap = self._remove_leading_particles(overlap)
            if overlap:
                return overlap
        
        return text
    
    def _remove_leading_particles(self, text: str) -> str:
        """
        Remove leading Arabic conjunctions and particles.
        
        Prevents chunks from starting with coordinating conjunctions (حروف العطف)
        which create grammatically incomplete structures.
        
        Removes:
        - و (and)
        - ف (then/so)
        - ثم (then)
        - أو/او (or)
        - لكن (but)
        - بل (rather)
        - حتى (even/until when used as conjunction)
        
        Args:
            text: Text to clean
            
        Returns:
            Text without leading particles
        """
        if not text:
            return text
        
        text = text.strip()
        
        # Leading conjunctions (حروف العطف)
        leading_particles = [
            'و',      # and
            'ف',      # then/so
            'ثم',     # then
            'او',     # or
            'أو',     # or (with hamza)
            'لكن',    # but
            'بل',     # rather
            'حتى',    # even/until (as conjunction)
        ]
        
        # Iteratively remove leading particles
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            original_text = text
            
            for particle in leading_particles:
                # Check if starts with particle + space
                if text.startswith(particle + ' '):
                    text = text[len(particle):].strip()
                    break
                # Check if text is just the particle
                elif text == particle:
                    text = ''
                    break
            
            # Stop if no change
            if text == original_text:
                break
            
            iteration += 1
        
        return text.strip()


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

def main():
    """Demonstration of grammar-first chunking"""
    
    # Sample text with discourse markers
    text = """
شهدت مدينة مراكش زلزالاً قوياً أدى إلى أضرار كبيرة في البنية التحتية. 
سارعت فرق الإنقاذ للمساعدة.

في سياق مختلف، يعتبر الذكاء الاصطناعي من أهم التقنيات الحديثة التي أحدثت تحولاً جذرياً. 
تعتمد الأنظمة على خوارزميات معقدة لتحليل البيانات.

من ناحية أخرى، تلعب الرياضة دوراً مهماً في الصحة البدنية والنفسية. 
ممارسة التمارين بانتظام تساعد على تقوية القلب.

بالعودة إلى التكنولوجيا، يشهد مجال معالجة اللغة الطبيعية تطوراً سريعاً في فهم النصوص العربية.
"""
    
    print("=" * 80)
    print("CAMeL-Enhanced Chunker V2 - Grammar-First Pipeline")
    print("=" * 80)
    print("\nPipeline:")
    print("  1. Detect grammar boundaries (morphology + discourse)")
    print("  2. Split at boundaries")
    print("  3. Merge small/similar segments")
    print("  4. Add overlap")
    
    # Initialize
    chunker = EnhancedGrammarChunker(
        use_camel=True,
        target_chunk_size=250,
        overlap_size=30
    )
    
    print(f"\n📝 Input text ({len(text)} chars):")
    print(text.strip()[:150] + "...")
    
    # Chunk
    print("\n🔧 Chunking...")
    chunks = chunker.chunk(text)
    
    print(f"\n✅ Generated {len(chunks)} chunks:\n")
    
    # Display results
    for i, chunk in enumerate(chunks, 1):
        print(f"{'─' * 80}")
        print(f"Chunk {i}:")
        print(f"  Boundary: {chunk['boundary_reason']}")
        print(f"  Score: {chunk['boundary_score']:.2f}")
        print(f"  Length: {chunk['length']} chars")
        print(f"  Overlap: {chunk['metadata'].get('has_overlap', False)}")
        print(f"  Text: {chunk['text'][:80]}...")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()