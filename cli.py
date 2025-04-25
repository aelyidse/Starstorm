#!/usr/bin/env python3
import argparse
import asyncio
from typing import Optional, List
from core.plugin_manager import PluginManager
from core.dependency_injector import DependencyInjector
from core.system_manager import SystemManager

class StarstormCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Starstorm CLI')
        self.subparsers = self.parser.add_subparsers(dest='command')
        self._setup_commands()
        self.system = SystemManager()
        self.injector = DependencyInjector(self.system)
        self.plugin_manager = PluginManager(self.injector)

    def _setup_commands(self):
        # Plugin commands
        plugin_parser = self.subparsers.add_parser('plugin', help='Plugin management')
        plugin_subparsers = plugin_parser.add_subparsers(dest='plugin_command')
        
        # List plugins
        list_parser = plugin_subparsers.add_parser('list', help='List available plugins')
        
        # Load plugin
        load_parser = plugin_subparsers.add_parser('load', help='Load a plugin')
        load_parser.add_argument('name', help='Plugin name')
        load_parser.add_argument('--config', help='Config file path')

        # System commands
        system_parser = self.subparsers.add_parser('system', help='System management')
        system_subparsers = system_parser.add_subparsers(dest='system_command')
        
        # Start system
        start_parser = system_subparsers.add_parser('start', help='Start the system')
        
        # Stop system
        stop_parser = system_subparsers.add_parser('stop', help='Stop the system')

    async def run(self, args: Optional[List[str]] = None):
        parsed = self.parser.parse_args(args)
        
        if parsed.command == 'plugin':
            if parsed.plugin_command == 'list':
                plugins = self.plugin_manager.list_plugins()
                print("Available plugins:")
                for name in plugins:
                    print(f"- {name}")
            
            elif parsed.plugin_command == 'load':
                print(f"Loading plugin: {parsed.name}")
                # TODO: Load config from file if provided
                plugin = self.plugin_manager.load_plugin(parsed.name)
                print(f"Plugin {parsed.name} loaded successfully")
        
        elif parsed.command == 'system':
            if parsed.system_command == 'start':
                print("Starting system...")
                await self.system.start()
                print("System started")
            
            elif parsed.system_command == 'stop':
                print("Stopping system...")
                await self.system.stop()
                print("System stopped")

def main():
    cli = StarstormCLI()
    asyncio.run(cli.run())

if __name__ == '__main__':
    main()