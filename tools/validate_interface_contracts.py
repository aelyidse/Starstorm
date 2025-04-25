#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional, Set, Tuple

# Add the project root to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.interface import ComponentInterface
from integration.interface_discovery import InterfaceDiscovery
from integration.interface_validation import InterfaceValidator
from integration.interface_compatibility import InterfaceCompatibilityChecker

class InterfaceContractValidator:
    """
    Validates interface implementations against their contracts and generates compliance reports.
    Combines discovery, validation, and compatibility checking into a single tool.
    """
    def __init__(self):
        self.discovery = InterfaceDiscovery()
        self.validator = InterfaceValidator()
        self.compatibility_checker = InterfaceCompatibilityChecker()
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        
    def discover_and_validate(self, package_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Discover interfaces in the specified package and validate all implementations.
        
        Args:
            package_path: Dot-notation path to the package to scan
            
        Returns:
            Dictionary mapping interface names to validation results
        """
        # Discover interfaces
        interfaces = self.discovery.discover_interfaces(package_path)
        
        # Validate each interface implementation
        for name, interface_class in interfaces.items():
            self.validation_results[name] = {
                'interface': name,
                'implementations': [],
                'orphaned': True,
                'compliance': 'N/A'
            }
            
            # Find implementations
            implementations = self.discovery.find_implementations(interface_class)
            
            if implementations:
                self.validation_results[name]['orphaned'] = False
                
                # Create an instance to validate against
                try:
                    interface_instance = interface_class()
                    
                    # Validate each implementation
                    for impl_path in implementations:
                        module_path, class_name = impl_path.rsplit('.', 1)
                        try:
                            module = __import__(module_path, fromlist=[class_name])
                            impl_class = getattr(module, class_name)
                            impl_instance = impl_class()
                            
                            # Validate implementation against interface
                            errors = interface_instance.validate_implementation(impl_instance)
                            
                            compliance = 'COMPLIANT' if not errors else 'NON-COMPLIANT'
                            
                            self.validation_results[name]['implementations'].append({
                                'name': impl_path,
                                'compliance': compliance,
                                'errors': errors
                            })
                        except Exception as e:
                            self.validation_results[name]['implementations'].append({
                                'name': impl_path,
                                'compliance': 'ERROR',
                                'errors': [str(e)]
                            })
                except Exception as e:
                    self.validation_results[name]['compliance'] = 'ERROR'
                    self.validation_results[name]['error'] = str(e)
            
            # Set overall compliance
            if self.validation_results[name]['orphaned']:
                self.validation_results[name]['compliance'] = 'ORPHANED'
            elif 'error' in self.validation_results[name]:
                self.validation_results[name]['compliance'] = 'ERROR'
            else:
                implementations = self.validation_results[name]['implementations']
                if all(impl['compliance'] == 'COMPLIANT' for impl in implementations):
                    self.validation_results[name]['compliance'] = 'COMPLIANT'
                else:
                    self.validation_results[name]['compliance'] = 'NON-COMPLIANT'
                    
        return self.validation_results
    
    def generate_report(self, output_format: str = 'text', output_file: Optional[str] = None) -> str:
        """
        Generate a compliance report in the specified format.
        
        Args:
            output_format: Format of the report ('text', 'json', or 'html')
            output_file: Optional file path to write the report to
            
        Returns:
            Report content as a string
        """
        if output_format == 'json':
            report = json.dumps(self.validation_results, indent=2)
        elif output_format == 'html':
            report = self._generate_html_report()
        else:  # text
            report = self._generate_text_report()
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                
        return report
    
    def _generate_text_report(self) -> str:
        """Generate a plain text compliance report."""
        report = "INTERFACE CONTRACT COMPLIANCE REPORT\n"
        report += "===================================\n\n"
        
        # Summary
        compliant = sum(1 for r in self.validation_results.values() if r['compliance'] == 'COMPLIANT')
        non_compliant = sum(1 for r in self.validation_results.values() if r['compliance'] == 'NON-COMPLIANT')
        orphaned = sum(1 for r in self.validation_results.values() if r['compliance'] == 'ORPHANED')
        error = sum(1 for r in self.validation_results.values() if r['compliance'] == 'ERROR')
        
        report += f"Total Interfaces: {len(self.validation_results)}\n"
        report += f"Compliant: {compliant}\n"
        report += f"Non-Compliant: {non_compliant}\n"
        report += f"Orphaned: {orphaned}\n"
        report += f"Error: {error}\n\n"
        
        # Detailed results
        for name, result in self.validation_results.items():
            report += f"Interface: {name}\n"
            report += f"Compliance: {result['compliance']}\n"
            
            if 'error' in result:
                report += f"Error: {result['error']}\n"
                
            if not result['orphaned']:
                report += "Implementations:\n"
                for impl in result['implementations']:
                    report += f"  - {impl['name']}: {impl['compliance']}\n"
                    if impl['errors']:
                        for error in impl['errors']:
                            report += f"    * {error}\n"
            
            report += "\n"
            
        return report
    
    def _generate_html_report(self) -> str:
        """Generate an HTML compliance report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interface Contract Compliance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .compliant { color: green; }
                .non-compliant { color: red; }
                .orphaned { color: orange; }
                .error { color: darkred; }
                .summary { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>Interface Contract Compliance Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
        """
        
        # Summary
        compliant = sum(1 for r in self.validation_results.values() if r['compliance'] == 'COMPLIANT')
        non_compliant = sum(1 for r in self.validation_results.values() if r['compliance'] == 'NON-COMPLIANT')
        orphaned = sum(1 for r in self.validation_results.values() if r['compliance'] == 'ORPHANED')
        error = sum(1 for r in self.validation_results.values() if r['compliance'] == 'ERROR')
        
        html += f"""
                <table>
                    <tr><th>Total Interfaces</th><td>{len(self.validation_results)}</td></tr>
                    <tr><th>Compliant</th><td class="compliant">{compliant}</td></tr>
                    <tr><th>Non-Compliant</th><td class="non-compliant">{non_compliant}</td></tr>
                    <tr><th>Orphaned</th><td class="orphaned">{orphaned}</td></tr>
                    <tr><th>Error</th><td class="error">{error}</td></tr>
                </table>
            </div>
            
            <h2>Interface Details</h2>
            <table>
                <tr>
                    <th>Interface</th>
                    <th>Compliance</th>
                    <th>Details</th>
                </tr>
        """
        
        # Interface details
        for name, result in self.validation_results.items():
            compliance_class = result['compliance'].lower()
            
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td class="{compliance_class}">{result['compliance']}</td>
                    <td>
            """
            
            if 'error' in result:
                html += f"<p>Error: {result['error']}</p>"
                
            if not result['orphaned']:
                html += "<ul>"
                for impl in result['implementations']:
                    impl_class = impl['compliance'].lower()
                    html += f'<li>{impl["name"]}: <span class="{impl_class}">{impl["compliance"]}</span>'
                    
                    if impl['errors']:
                        html += "<ul>"
                        for error in impl['errors']:
                            html += f"<li>{error}</li>"
                        html += "</ul>"
                        
                    html += "</li>"
                html += "</ul>"
                
            html += """
                    </td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        return html

def main():
    parser = argparse.ArgumentParser(description='Validate interface contracts and generate compliance reports')
    parser.add_argument('--package', '-p', type=str, default='core',
                        help='Root package to scan for interfaces (default: core)')
    parser.add_argument('--format', choices=['text', 'json', 'html'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--output', '-o', type=str, help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    validator = InterfaceContractValidator()
    
    print(f"Discovering and validating interfaces in package: {args.package}")
    validator.discover_and_validate(args.package)
    
    report = validator.generate_report(args.format, args.output)
    
    if not args.output:
        print(report)
    else:
        print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()