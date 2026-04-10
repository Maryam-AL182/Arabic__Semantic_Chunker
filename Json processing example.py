"""
JSON File Processing Example - Arabic Semantic Chunker V2

This example shows how to:
1. Load a JSON file with Arabic text
2. Process each document/page
3. Apply semantic chunking
4. Save chunked results back to JSON

Input format: JSON with pages/documents containing text
Output format: JSON with chunked text and metadata
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from arabic_semantic_chunker import GrammarAwareSemanticChunker
from camel_integration import EnhancedGrammarChunker


def load_json_file(filepath: str) -> Dict:
    """
    Load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Parsed JSON data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded JSON file: {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)


def chunk_json_pages(
    data: Dict,
    chunker: Any,
    method: str = 'grammar_aware'
) -> Dict:
    """
    Process JSON data and chunk each page's text.
    
    Args:
        data: JSON data with 'pages' array
        chunker: Chunker instance (GrammarAwareSemanticChunker or EnhancedGrammarChunker)
        method: Chunking method ('grammar_aware' or 'camel')
    
    Returns:
        Processed JSON with chunked text
    """
    if 'pages' not in data:
        print("❌ Error: JSON must contain 'pages' array")
        sys.exit(1)
    
    processed_data = {
        'metadata': {
            'chunking_method': method,
            'processed_at': datetime.now().isoformat(),
            'total_pages': len(data['pages']),
            'total_chunks': 0
        },
        'pages': []
    }
    
    print(f"\n🔧 Processing {len(data['pages'])} pages...")
    
    for page_idx, page in enumerate(data['pages'], 1):
        # Extract text from page
        if isinstance(page.get('text'), list):
            # Text is array - join elements
            full_text = ' '.join(page['text'])
        elif isinstance(page.get('text'), str):
            # Text is string - use directly
            full_text = page['text']
        else:
            print(f"⚠ Warning: Page {page_idx} has no text field, skipping...")
            continue
        
        # Skip empty text
        if not full_text or not full_text.strip():
            print(f"⚠ Warning: Page {page_idx} has empty text, skipping...")
            continue
        
        # Chunk the text
        if method == 'grammar_aware':
            chunks = chunker.chunk(full_text, respect_grammar=True, add_overlap=True)
            chunk_dicts = chunker.chunk_to_dict(chunks)
        else:  # camel
            chunks = chunker.chunk(full_text)
            chunk_dicts = chunks
        
        # Create page entry
        processed_page = {
            'page_index': page_idx,
            'url': page.get('url', ''),
            'title': page.get('title', ''),
            'timestamp': page.get('timestamp', ''),
            'original_text_length': len(full_text),
            'num_chunks': len(chunk_dicts),
            'chunks': chunk_dicts
        }
        
        processed_data['pages'].append(processed_page)
        processed_data['metadata']['total_chunks'] += len(chunk_dicts)
        
        # Progress indicator
        if page_idx % 10 == 0:
            print(f"  Processed {page_idx}/{len(data['pages'])} pages...")
    
    print(f"\n✓ Completed processing:")
    print(f"  Total pages: {processed_data['metadata']['total_pages']}")
    print(f"  Total chunks: {processed_data['metadata']['total_chunks']}")
    
    return processed_data


def save_chunked_json(data: Dict, output_filepath: str):
    """
    Save chunked data to JSON file.
    
    Args:
        data: Processed data with chunks
        output_filepath: Output file path
    """
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved chunked data to: {output_filepath}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        sys.exit(1)


def print_sample_chunks(data: Dict, num_samples: int = 3):
    """
    Print sample chunks for verification.
    
    Args:
        data: Processed data
        num_samples: Number of sample chunks to display
    """
    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)
    
    chunk_count = 0
    
    for page in data['pages']:
        if chunk_count >= num_samples:
            break
        
        for chunk_idx, chunk in enumerate(page['chunks']):
            if chunk_count >= num_samples:
                break
            
            print(f"\nPage {page['page_index']}, Chunk {chunk_idx + 1}:")
            print(f"  Title: {page['title'][:60]}...")
            print(f"  Length: {chunk.get('length', len(chunk.get('text', '')))} chars")
            
            if 'grammar_score' in chunk:
                print(f"  Grammar: {chunk['grammar_score']:.2f}")
                print(f"  Semantic: {chunk['semantic_score']:.2f}")
            elif 'boundary_score' in chunk:
                print(f"  Boundary: {chunk['boundary_reason']} ({chunk['boundary_score']:.2f})")
            
            chunk_text = chunk.get('text', '')
            print(f"  Text: {chunk_text[:100]}...")
            
            chunk_count += 1
    
    print("=" * 80)


def generate_statistics(data: Dict) -> Dict:
    """
    Generate statistics about the chunked data.
    
    Args:
        data: Processed data
    
    Returns:
        Statistics dictionary
    """
    import numpy as np
    
    all_chunk_lengths = []
    grammar_scores = []
    semantic_scores = []
    
    for page in data['pages']:
        for chunk in page['chunks']:
            chunk_length = chunk.get('length', len(chunk.get('text', '')))
            all_chunk_lengths.append(chunk_length)
            
            if 'grammar_score' in chunk:
                grammar_scores.append(chunk['grammar_score'])
                semantic_scores.append(chunk['semantic_score'])
    
    stats = {
        'total_chunks': len(all_chunk_lengths),
        'avg_chunk_length': np.mean(all_chunk_lengths) if all_chunk_lengths else 0,
        'std_chunk_length': np.std(all_chunk_lengths) if all_chunk_lengths else 0,
        'min_chunk_length': min(all_chunk_lengths) if all_chunk_lengths else 0,
        'max_chunk_length': max(all_chunk_lengths) if all_chunk_lengths else 0,
    }
    
    if grammar_scores:
        stats['avg_grammar_score'] = np.mean(grammar_scores)
        stats['avg_semantic_score'] = np.mean(semantic_scores)
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics in a formatted way"""
    print("\n" + "=" * 80)
    print("CHUNKING STATISTICS")
    print("=" * 80)
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Average chunk length: {stats['avg_chunk_length']:.1f} chars")
    print(f"  Std deviation: {stats['std_chunk_length']:.1f}")
    print(f"  Min chunk length: {stats['min_chunk_length']} chars")
    print(f"  Max chunk length: {stats['max_chunk_length']} chars")
    
    if 'avg_grammar_score' in stats:
        print(f"  Average grammar score: {stats['avg_grammar_score']:.2f}")
        print(f"  Average semantic score: {stats['avg_semantic_score']:.2f}")
    
    print("=" * 80)


# ==============================================================================
# MAIN PROCESSING FUNCTION
# ==============================================================================

def process_json_file(
    input_filepath: str,
    output_filepath: str = None,
    method: str = 'grammar_aware',
    target_chunk_size: int = 300,
    overlap_size: int = 40
):
    """
    Complete workflow to process JSON file with Arabic text chunking.
    
    Args:
        input_filepath: Path to input JSON file
        output_filepath: Path to output JSON file (optional)
        method: 'grammar_aware' or 'camel'
        target_chunk_size: Target size for chunks
        overlap_size: Overlap between chunks
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "JSON FILE PROCESSING EXAMPLE" + " " * 30 + "║")
    print("║" + " " * 25 + "Arabic Semantic Chunker V2" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Set default output filename
    if output_filepath is None:
        input_path = Path(input_filepath)
        output_filepath = str(input_path.parent / f"{input_path.stem}_chunked.json")
    
    print(f"\n📄 Input file: {input_filepath}")
    print(f"📄 Output file: {output_filepath}")
    print(f"🔧 Method: {method}")
    print(f"🎯 Target chunk size: {target_chunk_size} chars")
    print(f"🔗 Overlap: {overlap_size} chars")
    
    # Load JSON
    print("\n" + "─" * 80)
    data = load_json_file(input_filepath)
    
    # Initialize chunker
    print("\n🔧 Initializing chunker...")
    if method == 'grammar_aware':
        chunker = GrammarAwareSemanticChunker(
            target_chunk_size=target_chunk_size,
            overlap_size=overlap_size
        )
        print("✓ Using Grammar-Aware Semantic Chunker")
    elif method == 'camel':
        chunker = EnhancedGrammarChunker(
            use_camel=True,
            target_chunk_size=target_chunk_size,
            overlap_size=overlap_size
        )
        print("✓ Using CAMeL-Enhanced Grammar-First Chunker")
    else:
        print(f"❌ Error: Unknown method '{method}'")
        print("   Valid methods: 'grammar_aware', 'camel'")
        sys.exit(1)
    
    # Process pages
    print("\n" + "─" * 80)
    processed_data = chunk_json_pages(data, chunker, method)
    
    # Generate statistics
    stats = generate_statistics(processed_data)
    print_statistics(stats)
    
    # Save results
    print("\n" + "─" * 80)
    save_chunked_json(processed_data, output_filepath)
    
    # Show samples
    print_sample_chunks(processed_data, num_samples=5)
    
    print("\n✅ Processing complete!")
    print("=" * 80 + "\n")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

def main():
    """
    Example usage with different scenarios.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process JSON file with Arabic text chunking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with grammar-aware chunker
  python json_processing_example.py input.json
  
  # Process with CAMeL-enhanced chunker
  python json_processing_example.py input.json --method camel
  
  # Custom chunk size and overlap
  python json_processing_example.py input.json --size 500 --overlap 80
  
  # Specify output file
  python json_processing_example.py input.json -o output_chunks.json
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Path to input JSON file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to output JSON file (default: input_chunked.json)'
    )
    
    parser.add_argument(
        '-m', '--method',
        choices=['grammar_aware', 'camel'],
        default='grammar_aware',
        help='Chunking method (default: grammar_aware)'
    )
    
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=300,
        help='Target chunk size in characters (default: 300)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=40,
        help='Overlap size in characters (default: 40)'
    )
    
    args = parser.parse_args()
    
    # Process the file
    process_json_file(
        input_filepath=args.input_file,
        output_filepath=args.output,
        method=args.method,
        target_chunk_size=args.size,
        overlap_size=args.overlap
    )


if __name__ == "__main__":
    # If no command line args, show example with hardcoded file
    if len(sys.argv) == 1:
        print("\nNo command line arguments provided.")
        print("Running with example file: cleaned_for_maryam1.json\n")
        
        # Check if example file exists
        example_file = r"input.json"
        if Path(example_file).exists():
            process_json_file(
                input_filepath=example_file,
                output_filepath="ChunkedOutput.json",
                method='grammar_aware',
                target_chunk_size=300,
                overlap_size=40
            )
        else:
            print(f"Example file '{example_file}' not found.")
            print("\nUsage:")
            print("  python json_processing_example.py <input_file.json>")
            print("\nFor more options:")
            print("  python json_processing_example.py --help")
    else:
        main()
