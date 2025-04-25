#!/usr/bin/env python3
import os
import sys
import argparse
from integration.interface_discovery import InterfaceDiscovery
from integration.interface_documentation import InterfaceDocumentationGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate interface documentation')
    parser.add_argument('--format', choices=['markdown', 'json', 'html'], default='markdown',
                        help='Output format (default: markdown)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    parser.add_argument('--package', '-p', type=str, default='core',
                        help='Root package to scan for interfaces (default: core)')
    
    args = parser.parse_args()
    
    # Initialize discovery
    discovery = InterfaceDiscovery()
    print(f"Discovering interfaces in package: {args.package}")
    interfaces = discovery.discover_interfaces(args.package)
    print(f"Discovered {len(interfaces)} interfaces")
    
    # Generate documentation
    generator = InterfaceDocumentationGenerator(discovery)
    
    output_file = None
    if args.output:
        output_file = open(args.output, 'w')
    
    try:
        if args.format == 'markdown':
            content = generator.generate_markdown(output_file)
        elif args.format == 'json':
            content = generator.generate_json(output_file)
        elif args.format == 'html':
            content = generator.generate_html(output_file)
            
        if not args.output:
            print(content)
    finally:
        if output_file:
            output_file.close()
            print(f"Documentation written to {args.output}")

if __name__ == "__main__":
    main()