"""Setup script for Document Q&A Agent."""

import os
import sys
import subprocess
from pathlib import Path


def create_env_file():
    """Create .env file from template."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example_path.exists():
        print("📝 Creating .env file from .env.example")
        env_content = env_example_path.read_text()
        env_path.write_text(env_content)
        print("⚠️  Please edit .env and add your OpenAI API key")
        return False
    else:
        print("❌ .env.example not found")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["data", "uploads", "logs", "chroma_db"]
    
    print("\n📁 Creating directories...")
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✅ {dir_name}/")
    
    return True


def main():
    """Run setup process."""
    print("🚀 Document Q&A Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    create_directories()
    
    # Create .env file
    env_ready = create_env_file()
    
    # Install dependencies
    deps_installed = install_dependencies()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    
    if deps_installed and env_ready:
        print("\n✅ Setup complete! You're ready to go.")
        print("\nTo start the application:")
        print("  streamlit run app.py")
    elif deps_installed and not env_ready:
        print("\n⚠️  Setup partially complete.")
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: streamlit run app.py")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
    
    print("\nFor testing, run:")
    print("  python test_system.py")


if __name__ == "__main__":
    main()
