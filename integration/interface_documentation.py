from typing import Dict, Any, List, Optional, Type, TextIO
import json
import os
import markdown
from core.interface import ComponentInterface
from integration.interface_discovery import InterfaceDiscovery

class InterfaceDocumentationGenerator:
    """
    Generates documentation for interfaces in various formats.
    Supports markdown, JSON, and HTML output formats.
    """
    def __init__(self, discovery: Optional[InterfaceDiscovery] = None):
        self.discovery = discovery or InterfaceDiscovery()
        
    def generate_markdown(self, output_file: Optional[TextIO] = None) -> str:
        """
        Generate markdown documentation for all discovered interfaces.
        
        Args:
            output_file: Optional file-like object to write to
            
        Returns:
            Markdown string representation of interfaces
        """
        report = self.discovery.generate_interface_report()
        
        md_content = "# System Interface Documentation\n\n"
        
        # Summary section
        md_content += "## Interface Summary\n\n"
        md_content += f"Total Interfaces: {len(report['interfaces'])}\n\n"
        md_content += "| Interface | Implementation Count |\n"
        md_content += "|-----------|----------------------|\n"
        
        for name, count in report['implementation_count'].items():
            md_content += f"| {name} | {count} |\n"
        
        # Orphaned interfaces
        if report['orphaned_interfaces']:
            md_content += "\n## Orphaned Interfaces\n\n"
            md_content += "These interfaces have no implementations:\n\n"
            for name in report['orphaned_interfaces']:
                md_content += f"- {name}\n"
        
        # Detailed interface documentation
        md_content += "\n## Interface Details\n\n"
        
        for name, data in report['interfaces'].items():
            md_content += f"### {name}\n\n"
            
            if 'error' in data:
                md_content += f"Error: {data['error']}\n\n"
                continue
                
            description = data['description']
            md_content += f"Version: {description.get('version', 'Unknown')}\n\n"
            
            # Methods
            if 'methods' in description:
                md_content += "#### Methods\n\n"
                for method_name, method_info in description['methods'].items():
                    md_content += f"##### `{method_name}`\n\n"
                    md_content += f"Description: {method_info.get('description', 'No description')}\n\n"
                    md_content += f"Return Type: {method_info.get('return_type', 'None')}\n\n"
                    
                    if method_info.get('parameters'):
                        md_content += "Parameters:\n\n"
                        for param, param_type in method_info['parameters'].items():
                            md_content += f"- `{param}`: {param_type}\n"
                    md_content += "\n"
            
            # Properties
            if 'properties' in description:
                md_content += "#### Properties\n\n"
                for prop_name, prop_info in description['properties'].items():
                    md_content += f"##### `{prop_name}`\n\n"
                    md_content += f"Type: {prop_info.get('type', 'Unknown')}\n\n"
                    md_content += f"Description: {prop_info.get('description', 'No description')}\n\n"
                    md_content += f"Required: {prop_info.get('required', False)}\n\n"
                    md_content += f"Read Only: {prop_info.get('read_only', False)}\n\n"
            
            # Events
            if 'events' in description and description['events']:
                md_content += "#### Events\n\n"
                for event in description['events']:
                    md_content += f"- `{event}`\n"
                md_content += "\n"
            
            # Implementations
            if data['implementations']:
                md_content += "#### Implementations\n\n"
                for impl in data['implementations']:
                    md_content += f"- {impl}\n"
                md_content += "\n"
        
        if output_file:
            output_file.write(md_content)
            
        return md_content
    
    def generate_json(self, output_file: Optional[TextIO] = None) -> str:
        """
        Generate JSON documentation for all discovered interfaces.
        
        Args:
            output_file: Optional file-like object to write to
            
        Returns:
            JSON string representation of interfaces
        """
        report = self.discovery.generate_interface_report()
        json_content = json.dumps(report, indent=2)
        
        if output_file:
            output_file.write(json_content)
            
        return json_content
    
    def generate_html(self, output_file: Optional[TextIO] = None) -> str:
        """
        Generate HTML documentation for all discovered interfaces.
        
        Args:
            output_file: Optional file-like object to write to
            
        Returns:
            HTML string representation of interfaces
        """
        md_content = self.generate_markdown()
        html_content = markdown.markdown(md_content, extensions=['tables'])
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Interface Documentation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        if output_file:
            output_file.write(full_html)
            
        return full_html