"""
Complete Usage Examples - Arabic Semantic Chunker V2

Demonstrates:
1. Basic chunking
2. Grammar-aware chunking
3. CAMeL-enhanced chunking
4. Customization options
5. JSON export
6. RAG integration
7. Batch processing
"""

import json
import numpy as np
from pathlib import Path

from arabic_semantic_chunker import (
    GrammarAwareSemanticChunker,
    ArabicNormalizer,
    GrammarRuleEngine,
    ArabicEmbedder,
    Chunk
)

from camel_integration import EnhancedGrammarChunker


# ==============================================================================
# Sample Arabic Text
# ==============================================================================

SAMPLE_TEXT = """
ٱللُّغَةُ ٱلْعَرَبِيَّة هي أكثر اللغات السامية تحدثًا، وإحدى أكثر اللغات انتشاراً في العالم، يتحدثها أكثر من 467 مليون نسمة.ويتوزع متحدثوها في الوطن العربي، بالإضافة إلى العديد من المناطق الأخرى المجاورة كالأحواز وتركيا وتشاد ومالي والسنغال وإرتيريا وإثيوبيا وجنوب السودان وإيران. وبذلك فهي تحتل المركز الرابع أو الخامس من حيث اللغات الأكثر انتشارًا في العالم، وهي تحتل المركز الثالث تبعًا لعدد الدول التي تعترف بها كلغة رسمية؛ إذ تعترف بها 27 دولة لغةً رسميةً، واللغة الرابعة من حيث عدد المستخدمين على الإنترنت. اللغةُ العربيةُ ذات أهمية قصوى لدى المسلمين، فهي عندَهم لغةٌ مقدسة إذ إنها لغة القرآن، وهي لغةُ الصلاة وأساسيةٌ في القيام بالعديد من العبادات والشعائرِ الإسلامية. العربيةُ هي أيضاً لغة شعائرية رئيسية لدى عدد من الكنائس المسيحية في الوطن العربي، كما كُتبَت بها كثير من أهمِّ الأعمال الدينية والفكرية اليهودية في العصور الوسطى. ارتفعتْ مكانةُ اللغةِ العربية إثْرَ انتشارِ الإسلام بين الدول إذ أصبحت لغة السياسة والعلم والأدب لقرون طويلة في الأراضي التي حكمها المسلمون. وللغة العربية تأثير مباشر وغير مباشر على كثير من اللغات الأخرى في العالم الإسلامي، كالتركية والفارسية والأمازيغية والكردية والأردية والماليزية والإندونيسية والألبانية وبعض اللغات الإفريقية الأخرى مثل الهاوسا والسواحيلية والتجرية والأمهرية والصومالية، وبعض اللغات الأوروبية وخاصةً المتوسطية كالإسبانية والبرتغالية والمالطية والصقلية؛ ودخلت الكثير من مصطلحاتها في اللغة الإنجليزية واللغات الأخرى، مثل أدميرال والتعريفة والكحول والجبر وأسماء النجوم. كما أنها تُدرَّس بشكل رسمي أو غير رسمي في الدول الإسلامية والدول الإفريقية المحاذية للوطن العربي.
"""


# ==============================================================================
# Example 1: Basic Chunking
# ==============================================================================

def example_1_basic_chunking():
    """Basic semantic chunking without grammar constraints"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Semantic Chunking")
    print("=" * 80)
    
    # Initialize chunker
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=200,
        overlap_size=30
    )
    
    # Chunk text (no grammar constraints)
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=False, add_overlap=False)
    
    print(f"\n✓ Created {len(chunks)} chunks (semantic only)\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Preview: {chunk.text[:70]}...")
        print()


# ==============================================================================
# Example 2: Grammar-Aware Chunking
# ==============================================================================

def example_2_grammar_aware():
    """Chunking with grammar constraints"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Grammar-Aware Chunking")
    print("=" * 80)
    
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=200,
        overlap_size=30
    )
    
    # Chunk with grammar awareness
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=True)
    
    print(f"\n✓ Created {len(chunks)} chunks (grammar-aware)\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} [{chunk.type}]:")
        print(f"  Grammar score: {chunk.grammar_score:.2f}")
        print(f"  Semantic score: {chunk.semantic_score:.2f}")
        print(f"  Has overlap: {chunk.metadata.get('has_overlap', False)}")
        print(f"  Preview: {chunk.text[:70]}...")
        print()


# ==============================================================================
# Example 3: CAMeL-Enhanced Chunking with Grammar-Aware Overlap (Best Quality)
# ==============================================================================

def example_3_camel_enhanced():
    """
    CAMeL Tools enhanced chunking demonstrating the complete pipeline:
    
    PIPELINE STAGES:
    1. Grammar Analysis (CAMeL morphology + discourse markers)
    2. Semantic Similarity (embeddings for refinement)
    3. Grammar-Aware Overlap (intelligent boundary-respecting overlap)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Complete Grammar-First Pipeline with Intelligent Overlap")
    print("=" * 80)
    
    print("\nPipeline Stages:")
    print("  1️⃣  Grammar Analysis → Detect morphological boundaries")
    print("  2️⃣  Semantic Refinement → Merge similar segments")
    print("  3️⃣  Grammar-Aware Overlap → Add linguistically coherent overlap")
    print()
    
    # Initialize CAMeL-enhanced chunker
    chunker = EnhancedGrammarChunker(
        use_camel=True,  # Use CAMeL Tools for morphological analysis
        target_chunk_size=250,
        overlap_size=40  # Will create grammar-aware overlap
    )
    
    # Chunk the text (applies full pipeline)
    chunks = chunker.chunk(SAMPLE_TEXT)
    
    print(f"\n✓ Created {len(chunks)} chunks using complete pipeline\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Boundary: {chunk['boundary_reason']}")
        print(f"  Score: {chunk['boundary_score']:.2f}")
        print(f"  Length: {chunk['length']} chars")
        
        # Show overlap details
        if chunk['metadata'].get('has_overlap'):
            overlap_type = chunk['metadata'].get('overlap_type', 'character')
            overlap_size = chunk['metadata'].get('overlap_size', 0)
            print(f"  Overlap: {overlap_type} ({overlap_size} chars)")
        else:
            print(f"  Overlap: None (first chunk)")
        
        print(f"  Preview: {chunk['text'][:70]}...")
        print()
    
    # Demonstrate overlap quality
    print("─" * 80)
    print("OVERLAP QUALITY DEMONSTRATION")
    print("─" * 80)
    
    if len(chunks) >= 2:
        print("\n🔍 Examining overlap between Chunk 1 and Chunk 2:\n")
        
        chunk1_text = chunks[0]['text']
        chunk2_text = chunks[1]['text']
        
        # Extract the overlap portion
        if chunks[1]['metadata'].get('has_overlap'):
            overlap_size = chunks[1]['metadata'].get('overlap_size', 0)
            overlap_text = chunk2_text[:overlap_size]
            
            print(f"Chunk 1 ending:")
            print(f"  ...{chunk1_text[-100:]}\n")
            
            print(f"Overlap (grammar-aware, {overlap_size} chars):")
            print(f"  {overlap_text}\n")
            
            print(f"Chunk 2 beginning (after overlap):")
            print(f"  {chunk2_text[overlap_size:overlap_size+100]}...")
            
            print("\n✅ Notice: Overlap preserves complete linguistic units")
            print("   (sentences, phrases, or discourse markers)")
        
    print("\n" + "=" * 80)


# ==============================================================================
# Example 4: Customization
# ==============================================================================

def example_4_customization():
    """Customize chunker parameters and grammar rules"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Customization")
    print("=" * 80)
    
    # Custom chunker with larger chunks and more overlap
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=500,      # Larger chunks
        overlap_size=100,            # More overlap
        embedder_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Customize grammar engine
    print("\n🔧 Customizing grammar rules...")
    
    # Add custom relative pronouns
    chunker.grammar_engine.REL_PRONOUNS.add('ذلك')
    chunker.grammar_engine.REL_PRONOUNS.add('تلك')
    
    # Add custom prepositions
    chunker.grammar_engine.PREPOSITIONS.add('خلال')
    chunker.grammar_engine.PREPOSITIONS.add('حول')
    chunker.grammar_engine.PREPOSITIONS.add('ضد')
    
    print(f"  Relative pronouns: {len(chunker.grammar_engine.REL_PRONOUNS)}")
    print(f"  Prepositions: {len(chunker.grammar_engine.PREPOSITIONS)}")
    
    # Chunk with custom settings
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=True)
    
    print(f"\n✓ Created {len(chunks)} chunks with custom settings")
    print(f"  Average size: {np.mean([len(c.text) for c in chunks]):.1f} chars")


# ==============================================================================
# Example 5: JSON Export
# ==============================================================================

def example_5_json_export():
    """Export chunks to JSON format"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: JSON Export")
    print("=" * 80)
    
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=300,
        overlap_size=40
    )
    
    # Chunk text
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=True)
    
    # Convert to dict
    chunks_dict = chunker.chunk_to_dict(chunks)
    
    # Save to JSON
    output_file = "example_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved {len(chunks)} chunks to {output_file}")
    
    # Display sample
    print("\nSample chunk structure:")
    print(json.dumps(chunks_dict[0], ensure_ascii=False, indent=2)[:300] + "...")


# ==============================================================================
# Example 6: Batch Processing
# ==============================================================================

def example_6_batch_processing():
    """Process multiple documents"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Batch Processing")
    print("=" * 80)
    
    # Multiple documents
    documents = [
        "هذا هو المستند الأول. يحتوي على معلومات حول الذكاء الاصطناعي.",
        "المستند الثاني يتحدث عن التكنولوجيا. التطورات الحديثة مثيرة للاهتمام.",
        SAMPLE_TEXT[:300]  # Portion of sample text
    ]
    
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=150,
        overlap_size=20
    )
    
    all_chunks = []
    
    print(f"\n🔧 Processing {len(documents)} documents...")
    
    for doc_id, doc in enumerate(documents, 1):
        chunks = chunker.chunk(doc, respect_grammar=True, add_overlap=False)
        
        # Add document metadata
        for chunk_idx, chunk in enumerate(chunks):
            chunk_dict = chunk.to_dict()
            chunk_dict['document_id'] = doc_id
            chunk_dict['chunk_id_in_doc'] = chunk_idx
            all_chunks.append(chunk_dict)
    
    print(f"\n✓ Processed {len(documents)} documents")
    print(f"✓ Generated {len(all_chunks)} total chunks")
    
    # Save all chunks
    with open("batch_chunks.json", 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to batch_chunks.json")


# ==============================================================================
# Example 7: RAG Integration
# ==============================================================================

def example_7_rag_integration():
    """Integrate with RAG system"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: RAG System Integration")
    print("=" * 80)
    
    # Step 1: Chunk documents
    print("\n1️⃣  Document Processing:")
    
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=300,
        overlap_size=50
    )
    
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=True)
    print(f"   ✓ Created {len(chunks)} semantic chunks")
    print(f"   ✓ All chunks have embeddings")
    
    # Step 2: Simulate vector database storage
    print("\n2️⃣  Vector Database Storage:")
    
    vector_db = []
    for i, chunk in enumerate(chunks):
        vector_db.append({
            'id': i,
            'text': chunk.text,
            'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None,
            'metadata': {
                'type': chunk.type,
                'grammar_score': chunk.grammar_score,
                'semantic_score': chunk.semantic_score,
                'length': len(chunk.text)
            }
        })
    
    print(f"   ✓ Stored {len(vector_db)} chunks")
    
    # Step 3: Query simulation
    print("\n3️⃣  Query Processing:")
    
    query = "ما هو الذكاء الاصطناعي؟"
    print(f"   Query: {query}")
    
    # Get query embedding
    embedder = ArabicEmbedder()
    query_embedding = embedder.embed(query)
    print("   ✓ Generated query embedding")
    
    # Search (cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = []
    for item in vector_db:
        if item['embedding'] is not None:
            db_emb = np.array(item['embedding'])
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                db_emb.reshape(1, -1)
            )[0][0]
            similarities.append((item['id'], sim, item['text'], item['metadata']))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:3]
    
    print(f"\n4️⃣  Top {len(top_results)} Results:")
    for rank, (chunk_id, score, text, metadata) in enumerate(top_results, 1):
        print(f"\n   {rank}. Chunk {chunk_id} (similarity: {score:.3f})")
        print(f"      Grammar: {metadata['grammar_score']:.2f}")
        print(f"      Text: {text[:80]}...")


# ==============================================================================
# Example 8: Quality Metrics
# ==============================================================================

def example_8_quality_metrics():
    """Analyze chunk quality metrics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Quality Metrics")
    print("=" * 80)
    
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=300,
        overlap_size=40
    )
    
    chunks = chunker.chunk(SAMPLE_TEXT, respect_grammar=True)
    
    # Calculate metrics
    lengths = [len(c.text) for c in chunks]
    grammar_scores = [c.grammar_score for c in chunks]
    semantic_scores = [c.semantic_score for c in chunks]
    
    print(f"\n📊 Chunking Statistics:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Average length: {np.mean(lengths):.1f} chars")
    print(f"   Min length: {min(lengths)} chars")
    print(f"   Max length: {max(lengths)} chars")
    print(f"   Length std dev: {np.std(lengths):.1f}")
    print(f"   Size variance: {np.std(lengths) / np.mean(lengths):.2f}")
    print()
    print(f"   Average grammar score: {np.mean(grammar_scores):.2f}")
    print(f"   Average semantic score: {np.mean(semantic_scores):.2f}")
    print(f"   Chunks with high grammar score (>0.8): {sum(1 for s in grammar_scores if s > 0.8)}")
    
    # Check discourse marker alignment
    discourse_markers = ['في سياق مختلف', 'من ناحية أخرى', 'بالعودة إلى']
    discourse_chunks = 0
    
    for chunk in chunks:
        chunk_start = chunk.text[:80].lower()
        if any(marker in chunk_start for marker in discourse_markers):
            discourse_chunks += 1
    
    print(f"   Chunks starting with discourse markers: {discourse_chunks}")


# ==============================================================================
# Example 9: Comparison of Strategies
# ==============================================================================

def example_9_strategy_comparison():
    """Compare different chunking strategies"""
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Strategy Comparison")
    print("=" * 80)
    
    # Strategy 1: Semantic only
    chunker1 = GrammarAwareSemanticChunker(target_chunk_size=300, overlap_size=0)
    chunks_semantic = chunker1.chunk(SAMPLE_TEXT, respect_grammar=False, add_overlap=False)
    
    # Strategy 2: Grammar-aware semantic
    chunks_grammar = chunker1.chunk(SAMPLE_TEXT, respect_grammar=True, add_overlap=False)
    
    # Strategy 3: CAMeL-enhanced
    chunker2 = EnhancedGrammarChunker(use_camel=True, target_chunk_size=300, overlap_size=0)
    chunks_camel = chunker2.chunk(SAMPLE_TEXT)
    
    print(f"\n📋 Comparison:")
    print(f"   Semantic only:         {len(chunks_semantic)} chunks")
    print(f"   Grammar-aware:         {len(chunks_grammar)} chunks")
    print(f"   CAMeL-enhanced:        {len(chunks_camel)} chunks")
    
    print(f"\n   Average sizes:")
    print(f"   Semantic only:         {np.mean([len(c.text) for c in chunks_semantic]):.1f} chars")
    print(f"   Grammar-aware:         {np.mean([len(c.text) for c in chunks_grammar]):.1f} chars")
    print(f"   CAMeL-enhanced:        {np.mean([c['length'] for c in chunks_camel]):.1f} chars")


# ==============================================================================
# Example 10: Advanced Customization
# ==============================================================================

def example_10_advanced_customization():
    """Advanced customization options"""
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Advanced Customization")
    print("=" * 80)
    
    # Create normalizer with custom settings
    normalizer = ArabicNormalizer(preserve_ta_marbuta=True)
    
    # Test normalization
    test_text = "مُحَمَّدٌ يَدْرُسُ فِي الـــمَدْرَسَةِ"
    normalized = normalizer.normalize(test_text)
    
    print(f"\nNormalization example:")
    print(f"  Original:   {test_text}")
    print(f"  Normalized: {normalized}")
    
    # Create custom grammar engine
    grammar_engine = GrammarRuleEngine()
    
    # Add domain-specific terms
    grammar_engine.REL_PRONOUNS.update(['هذا', 'ذلك', 'تلك'])
    grammar_engine.PREPOSITIONS.update(['خلال', 'حول', 'ضد', 'نحو'])
    
    print(f"\nGrammar engine customized:")
    print(f"  Relative pronouns: {len(grammar_engine.REL_PRONOUNS)}")
    print(f"  Prepositions: {len(grammar_engine.PREPOSITIONS)}")
    print(f"  Conjunctions: {len(grammar_engine.CLAUSE_CONNECTORS)}")


# ==============================================================================
# Example 11: JSON File Processing
# ==============================================================================

def example_11_json_file_processing():
    """Process JSON file with Arabic text pages"""
    print("\n" + "=" * 80)
    print("EXAMPLE 11: JSON File Processing")
    print("=" * 80)
    
    # Sample JSON structure (like cleaned_for_maryam1.json)
    sample_json = {
        "pages": [
            {
                "url": "http://example.com/page1",
                "title": "إعلان المنح الدراسية",
                "timestamp": "2026-03-02T10:35:03.831871+00:00",
                "text": [
                    "اعلن زار تعليم عالي بحث علمي منح دراسي مقدمة معهد وطني تعليم دولي. "
                    "مزية منحة اعفاء رسم دراسي سنة لغة كوري مجانية تذكرة سفر درجة اقتصادي. "
                    "معلومة مفصل شرط اهلي تقديم مستند مطلوب توفر موقع رسمي سفارة."
                ]
            },
            {
                "url": "http://example.com/page2",
                "title": "الهيئة الوطنية للجودة",
                "timestamp": "2026-03-02T10:35:09.466227+00:00",
                "text": [
                    "صدر تعميم موعد دورة اولي امتحان وطنية موحد اختصاص طب بشري طب سن صيدلة. "
                    "امتحان شفهي عملية اقام اسبوع ولي امتحان نظري طالب حقق نجاح جزء نظري."
                ]
            }
        ]
    }
    
    print("\n📝 Sample JSON structure:")
    print(f"  Pages: {len(sample_json['pages'])}")
    
    # Initialize chunker
    chunker = GrammarAwareSemanticChunker(
        target_chunk_size=150,
        overlap_size=30
    )
    
    # Process each page
    processed_pages = []
    
    for page_idx, page in enumerate(sample_json['pages'], 1):
        # Join text array
        full_text = ' '.join(page['text'])
        
        # Chunk the text
        chunks = chunker.chunk(full_text, respect_grammar=True)
        chunk_dicts = chunker.chunk_to_dict(chunks)
        
        # Store processed page
        processed_page = {
            'page_index': page_idx,
            'url': page['url'],
            'title': page['title'],
            'num_chunks': len(chunk_dicts),
            'chunks': chunk_dicts
        }
        
        processed_pages.append(processed_page)
        
        print(f"\n  Page {page_idx}: {page['title'][:40]}...")
        print(f"    Original text: {len(full_text)} chars")
        print(f"    Generated: {len(chunk_dicts)} chunks")
    
    # Save to JSON
    output_data = {
        'metadata': {
            'total_pages': len(processed_pages),
            'total_chunks': sum(p['num_chunks'] for p in processed_pages)
        },
        'pages': processed_pages
    }
    
    output_file = "sample_chunked_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved chunked JSON to: {output_file}")
    print(f"  Total pages: {output_data['metadata']['total_pages']}")
    print(f"  Total chunks: {output_data['metadata']['total_chunks']}")
    
    # Show sample chunk
    if processed_pages and processed_pages[0]['chunks']:
        sample_chunk = processed_pages[0]['chunks'][0]
        print(f"\n📄 Sample chunk:")
        print(f"  Length: {sample_chunk['length']} chars")
        print(f"  Grammar: {sample_chunk['grammar_score']:.2f}")
        print(f"  Text: {sample_chunk['text'][:80]}...")


# ==============================================================================
# Example 12: Overlap Strategy Comparison
# ==============================================================================

def example_12_overlap_comparison():
    """
    Compare different overlap strategies:
    1. Character-based overlap (naive)
    2. Grammar-aware overlap (GrammarAwareSemanticChunker)
    3. Morphology-aware overlap (EnhancedGrammarChunker with CAMeL)
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 12: Overlap Strategy Comparison")
    print("=" * 80)
    
    test_text = """
التعليم هو حجر الزاوية في بناء المجتمعات المتقدمة. يساهم في تطوير المهارات وتنمية القدرات الفكرية للأفراد.

من ناحية أخرى، تلعب التكنولوجيا دوراً محورياً في تحسين جودة التعليم. توفر أدوات مبتكرة للتعلم عن بعد والوصول إلى الموارد التعليمية.

بالإضافة إلى ذلك، يعتبر البحث العلمي ركيزة أساسية للتقدم. يسهم في حل المشكلات المعقدة وتطوير حلول مبتكرة.
"""
    
    print("\n📝 Testing with sample text about education and technology\n")
    
    # Strategy 1: Grammar-Aware Overlap (Sentence-based)
    print("─" * 80)
    print("STRATEGY 1: Grammar-Aware Overlap")
    print("─" * 80)
    
    chunker1 = GrammarAwareSemanticChunker(
        target_chunk_size=150,
        overlap_size=40
    )
    
    chunks1 = chunker1.chunk(test_text, respect_grammar=True, add_overlap=True)
    
    print(f"\nGenerated {len(chunks1)} chunks")
    
    if len(chunks1) >= 2:
        print("\n🔍 Examining overlap between chunks:\n")
        for i in range(1, min(3, len(chunks1))):
            prev_text = chunks1[i-1].text
            curr_text = chunks1[i].text
            
            if chunks1[i].metadata.get('has_overlap'):
                overlap_size = chunks1[i].metadata.get('overlap_size', 0)
                overlap_text = curr_text[:overlap_size]
                
                print(f"Chunk {i-1} → Chunk {i}:")
                print(f"  Previous ending: ...{prev_text[-60:]}")
                print(f"  Overlap ({overlap_size} chars): [{overlap_text}]")
                print(f"  Current start: {curr_text[overlap_size:overlap_size+60]}...")
                print()
    
    # Strategy 2: Morphology-Aware Overlap (CAMeL-enhanced)
    print("─" * 80)
    print("STRATEGY 2: Morphology-Aware Overlap (CAMeL)")
    print("─" * 80)
    
    chunker2 = EnhancedGrammarChunker(
        use_camel=True,
        target_chunk_size=150,
        overlap_size=40
    )
    
    chunks2 = chunker2.chunk(test_text)
    
    print(f"\nGenerated {len(chunks2)} chunks")
    
    if len(chunks2) >= 2:
        print("\n🔍 Examining overlap between chunks:\n")
        for i in range(1, min(3, len(chunks2))):
            prev_text = chunks2[i-1]['text']
            curr_text = chunks2[i]['text']
            
            if chunks2[i]['metadata'].get('has_overlap'):
                overlap_size = chunks2[i]['metadata'].get('overlap_size', 0)
                overlap_type = chunks2[i]['metadata'].get('overlap_type', 'unknown')
                overlap_text = curr_text[:overlap_size]
                
                print(f"Chunk {i-1} → Chunk {i}:")
                print(f"  Type: {overlap_type}")
                print(f"  Previous ending: ...{prev_text[-60:]}")
                print(f"  Overlap ({overlap_size} chars): [{overlap_text}]")
                print(f"  Current start: {curr_text[overlap_size:overlap_size+60]}...")
                print()
    
    # Quality Analysis
    print("─" * 80)
    print("QUALITY ANALYSIS")
    print("─" * 80)
    
    print("\n✅ Grammar-Aware Overlap Benefits:")
    print("  • Respects sentence boundaries")
    print("  • Preserves complete phrases")
    print("  • Avoids mid-word cuts")
    
    print("\n✅ Morphology-Aware Overlap Benefits:")
    print("  • Detects syntactic units (verb/noun phrases)")
    print("  • Recognizes discourse markers")
    print("  • Respects morphological boundaries")
    print("  • Optimal for Arabic with complex morphology")
    
    print("\n💡 Recommendation:")
    print("  Use morphology-aware overlap (CAMeL) for best results")
    print("  Falls back to grammar-aware if CAMeL unavailable")
    
    print("\n" + "=" * 80)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ARABIC SEMANTIC CHUNKER V2" + " " * 32 + "║")
    print("║" + " " * 25 + "Usage Examples" + " " * 39 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        example_1_basic_chunking()
        example_2_grammar_aware()
        example_3_camel_enhanced()
        example_4_customization()
        example_5_json_export()
        example_6_batch_processing()
        example_7_rag_integration()
        example_8_quality_metrics()
        example_9_strategy_comparison()
        example_10_advanced_customization()
        example_11_json_file_processing()
        example_12_overlap_comparison()  # NEW: Grammar-aware overlap demonstration
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)
        
        print("\n💡 Key Takeaways:")
        print("  • Use respect_grammar=True for better linguistic quality")
        print("  • CAMeL-enhanced chunker provides best results")
        print("  • Grammar-aware overlap preserves linguistic coherence")
        print("  • Morphology-aware overlap (CAMeL) is optimal for Arabic")
        print("  • Adjust target_chunk_size based on your use case")
        print("  • Overlap improves retrieval in RAG systems")
        print("  • Monitor grammar and semantic scores for quality")
        print("  • For JSON files, use json_processing_example.py")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()