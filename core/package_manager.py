from typing import Dict, List, Optional
import importlib
import pkgutil
from pathlib import Path
from .component_registry import ComponentRegistry

class PackageManager:
    """
    Manages SDK component packages including installation, loading, and dependency resolution.
    """
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.installed_packages: Dict[str, Dict[str, str]] = {}
        self.package_paths: List[str] = []
        self._dependency_graph = nx.DiGraph()  # NetworkX graph for tracking dependencies
        
    def add_package_path(self, path: str) -> None:
        """Add a path to search for packages"""
        if path not in self.package_paths:
            self.package_paths.append(path)
            
    def install_package(self, package_name: str, version: str = "latest") -> bool:
        """Install a package and register its components"""
        try:
            package = importlib.import_module(package_name)
            package_info = {
                'version': version,
                'path': str(Path(package.__file__).parent),
                'dependencies': getattr(package, '__dependencies__', [])
            }
            self.installed_packages[package_name] = package_info
            
            # Add to dependency graph
            self._dependency_graph.add_node(package_name)
            for dep in package_info['dependencies']:
                self._dependency_graph.add_edge(package_name, dep)
            
            # Check for missing dependencies
            missing_deps = [d for d in package_info['dependencies'] 
                          if d not in self.installed_packages]
            if missing_deps:
                raise ImportError(f"Missing dependencies: {missing_deps}")
                
            # Scan package for components
            self.registry.scan_package(package_name)
            return True
        except ImportError as e:
            raise ImportError(f"Failed to install package {package_name}: {str(e)}")
            
    def resolve_dependencies(self, package_name: str) -> List[str]:
        """Resolve all dependencies (direct and indirect) for a package"""
        if package_name not in self._dependency_graph:
            return []
            
        dependencies = set()
        for node in self._dependency_graph:
            if node == package_name:
                continue
            try:
                if nx.has_path(self._dependency_graph, package_name, node):
                    dependencies.add(node)
            except:
                pass
        return sorted(dependencies)
        
    def validate_dependencies(self) -> bool:
        """Check if all dependencies are satisfied"""
        for package in self.installed_packages:
            for dep in self.installed_packages[package]['dependencies']:
                if dep not in self.installed_packages:
                    return False
        return True
        
    def get_dependency_tree(self) -> Dict[str, List[str]]:
        """Get the complete dependency tree"""
        return {pkg: self.resolve_dependencies(pkg) 
               for pkg in self.installed_packages}
    def get_package_info(self, package_name: str) -> Optional[Dict[str, str]]:
        """Get information about an installed package"""
        return self.installed_packages.get(package_name)
        
    def list_components(self, package_name: str) -> List[str]:
        """List all components from a package"""
        return [
            name for name, component in self.registry.get_all_components().items()
            if component.__module__.startswith(package_name)
        ]
    
    def check_for_updates(self, package_name: str) -> Dict[str, str]:
        """
        Check for available updates for a package.
        
        Args:
            package_name: Name of package to check
            
        Returns:
            Dictionary containing current and available versions
        """
        if package_name not in self.installed_packages:
            raise ValueError(f"Package {package_name} not installed")
            
        current_version = self.installed_packages[package_name]['version']
        
        # In a real implementation, this would query a package registry
        # For now, we'll simulate checking for updates
        available_version = "1.1.0" if current_version == "1.0.0" else current_version
        
        return {
            'current': current_version,
            'available': available_version,
            'update_available': available_version != current_version
        }
        
    def update_package(self, package_name: str, version: str = "latest") -> bool:
        """
        Update an installed package to a specified version.
        
        Args:
            package_name: Name of package to update
            version: Version to update to (defaults to latest)
            
        Returns:
            True if update succeeded, False otherwise
        """
        if package_name not in self.installed_packages:
            raise ValueError(f"Package {package_name} not installed")
            
        # First unregister all components from this package
        components = self.list_components(package_name)
        for component_name in components:
            self.registry.unregister_component(component_name)
            
        # Then reinstall the package
        try:
            return self.install_package(package_name, version)
        except ImportError as e:
            # If update fails, try to reinstall original version
            original_version = self.installed_packages[package_name]['version']
            self.install_package(package_name, original_version)
            raise ImportError(f"Update failed: {str(e)}. Reverted to version {original_version}")
            
    def batch_update(self, packages: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Update multiple packages at once with dependency resolution.
        
        Args:
            packages: List of package names to update
            
        Returns:
            Dictionary mapping package names to update results
        """
        results = {}
        
        # Resolve update order based on dependencies
        update_order = []
        for pkg in packages:
            update_order.extend(self.resolve_dependencies(pkg))
        update_order = list(dict.fromkeys(update_order + packages))  # Remove duplicates
        
        for pkg in update_order:
            try:
                if pkg not in self.installed_packages:
                    continue
                    
                update_info = self.check_for_updates(pkg)
                if update_info['update_available']:
                    success = self.update_package(pkg, update_info['available'])
                    results[pkg] = {
                        'status': 'success' if success else 'failed',
                        'version': update_info['available']
                    }
                else:
                    results[pkg] = {
                        'status': 'up_to_date',
                        'version': update_info['current']
                    }
            except Exception as e:
                results[pkg] = {
                    'status': 'error',
                    'error': str(e)
                }
                
        return results