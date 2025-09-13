#!/usr/bin/env python3
"""
PDF Text Extraction Module

This module provides functionality to extract text from PDF files and save them as plain text.
It's designed to be the first step in a RAG (Retrieval-Augmented Generation) pipeline.

Key Features:
- Extracts text from PDF pages using pdfplumber
- Handles pages with no extractable text (scanned images)
- Saves extracted text to organized output directory
- Provides detailed logging and progress information

Usage:
    python extract_pdf.py <pdf_file> [--output-dir <directory>]
    
Example:
    python extract_pdf.py hr_policy.pdf --output-dir docs
"""

import pdfplumber
import os
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple


class PDFTextExtractor:
    """
    A class to handle PDF text extraction with comprehensive error handling and logging.
    """
    
    def __init__(self, output_dir: str = "docs"):
        """
        Initialize the PDF text extractor.
        
        Args:
            output_dir (str): Directory to save extracted text files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[List[str], int, int]:
        """
        Extract text from all pages of a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Tuple[List[str], int, int]: (extracted_pages, total_pages, pages_with_text)
        """
        print(f"📄 Opening PDF file: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"📊 Total pages in PDF: {total_pages}")
                
                extracted_pages = []
                pages_with_text = 0
                
                # Process each page individually
                for page_num, page in enumerate(pdf.pages, start=1):
                    print(f"🔄 Processing page {page_num}/{total_pages}...", end=" ")
                    
                    # Extract text from the current page
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        # Page has extractable text
                        extracted_pages.append(page_text.strip())
                        pages_with_text += 1
                        print("✅ Text extracted")
                    else:
                        # Page has no extractable text (likely scanned image)
                        extracted_pages.append(f"[Page {page_num} — No extractable text (possibly scanned image)]")
                        print("⚠️  No text found (scanned image?)")
                
                print(f"\n📈 Extraction Summary:")
                print(f"   • Total pages processed: {total_pages}")
                print(f"   • Pages with text: {pages_with_text}")
                print(f"   • Pages without text: {total_pages - pages_with_text}")
                
                return extracted_pages, total_pages, pages_with_text
                
        except FileNotFoundError:
            print(f"❌ Error: PDF file not found: {pdf_path}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading PDF file: {str(e)}")
            sys.exit(1)
    
    def save_extracted_text(self, pages: List[str], pdf_filename: str) -> str:
        """
        Save extracted text pages to a single text file.
        
        Args:
            pages (List[str]): List of extracted page texts
            pdf_filename (str): Original PDF filename (without extension)
            
        Returns:
            str: Path to the saved text file
        """
        # Create output filename based on PDF name
        output_filename = f"{pdf_filename}.txt"
        output_path = self.output_dir / output_filename
        
        print(f"💾 Saving extracted text to: {output_path}")
        
        # Combine all pages with clear separators
        combined_text = "\n\n" + "="*80 + "\n\n".join(pages) + "\n\n" + "="*80
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            print(f"✅ Successfully saved {len(pages)} pages to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error saving text file: {str(e)}")
            sys.exit(1)
    
    def extract_pdf_to_txt(self, pdf_path: str) -> str:
        """
        Main method to extract text from PDF and save to text file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Path to the saved text file
        """
        print("🚀 Starting PDF text extraction process...")
        print("="*60)
        
        # Validate input file
        if not os.path.exists(pdf_path):
            print(f"❌ Error: PDF file does not exist: {pdf_path}")
            sys.exit(1)
        
        # Get PDF filename without extension
        pdf_filename = Path(pdf_path).stem
        
        # Step 1: Extract text from all pages
        print("\n📖 Step 1: Extracting text from PDF pages...")
        pages, total_pages, pages_with_text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Save extracted text to file
        print("\n💾 Step 2: Saving extracted text to file...")
        output_path = self.save_extracted_text(pages, pdf_filename)
        
        # Step 3: Display summary
        print("\n📋 Step 3: Extraction complete!")
        print("="*60)
        print(f"✅ PDF processed: {pdf_path}")
        print(f"✅ Text saved to: {output_path}")
        print(f"✅ Pages processed: {total_pages}")
        print(f"✅ Pages with text: {pages_with_text}")
        print(f"✅ Output directory: {self.output_dir}")
        
        return output_path


def main():
    """
    Main function to handle command line arguments and run the extraction process.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files and save as plain text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pdf.py hr_policy.pdf
  python extract_pdf.py hr_policy.pdf --output-dir documents
  python extract_pdf.py /path/to/file.pdf --output-dir /path/to/output
        """
    )
    
    # Add command line arguments
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to extract text from"
    )
    
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Output directory to save extracted text files (default: docs)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create extractor instance and run extraction
    extractor = PDFTextExtractor(output_dir=args.output_dir)
    output_path = extractor.extract_pdf_to_txt(args.pdf_path)
    
    print(f"\n🎉 Extraction completed successfully!")
    print(f"📁 Next step: Use the extracted text file for chunking and indexing")


if __name__ == "__main__":
    main()