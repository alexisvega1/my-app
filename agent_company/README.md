# Multi-Agent System v0.3

A sophisticated multi-agent system that implements autonomous code generation and tool building through a collaborative agent pipeline.

## 🏗️ Architecture

The system consists of 7 core components working in sequence:

```
User Input → Project Manager → Planner-2 → Code Builder → Critic → Tool Registry → Executor
                                    ↓
                                Memory (FAISS)
```

### Components

1. **Project Manager** - Manages backlog and ticket creation
2. **Planner-2** - Decides on actions based on patterns and memory
3. **Code Builder** - Generates tool specifications and code
4. **Critic** - Reviews and validates code for quality and safety
5. **Tool Registry** - Manages dynamic tool registration and loading
6. **Executor** - Runs tests and executes tools safely
7. **Memory** - FAISS-based vector storage for experience learning

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd agent_company

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the System

```bash
python main.py
```

### Example Usage

```
🤖 Multi-Agent System > Build a function that fetches current ETH-USD price and returns it as float

📋 PM: Added ticket abc123 to backlog
🧠 Planner-2: Created plan: build_spec
🔨 Code-Builder: Generated spec for fetch_eth_price
🔍 Critic: Review complete - approved
📚 Tool-Registry: Registered fetch_eth_price
⚡ Executor: Tool execution - success
🧠 Memory: Stored experience

✅ Successfully built and loaded: fetch_eth_price
📝 Description: Build a function that fetches current ETH-USD price and returns it as float
🔧 Output: 2345.67
```

## 🔧 How It Works

### 1. Project Manager
- Creates tickets from user input
- Manages backlog with priorities and status
- Persists data to `backlog.json`

### 2. Planner-2
- Uses pattern matching to determine actions
- Consults memory for similar experiences
- Supports actions: `build_spec`, `search_info`, `list_tools`, `help`

### 3. Code Builder
- Analyzes requirements to determine tool type
- Generates function names, parameters, and implementations
- Creates tool files in `dyn_tools/` directory
- Supports tool types: `data_fetch`, `calculation`, `utility`

### 4. Critic
- Performs safety checks for dangerous patterns
- Validates code quality and structure
- Provides feedback and suggestions
- Must approve before execution

### 5. Tool Registry
- Manages tool registration and loading
- Tracks usage statistics
- Supports hot-reloading of tools
- Persists registry to `tool_registry.json`

### 6. Executor
- Creates and runs test files
- Executes tools safely with timeout protection
- Validates tool safety before execution
- Generates test cases automatically

### 7. Memory
- Uses FAISS for vector similarity search
- Stores complete experiences (ticket → plan → spec → review → result)
- Enables learning from past successes and failures
- Persists to `memory.json`

## 📁 Project Structure

```
agent_company/
├── main.py              # Main entry point
├── project_manager.py   # Backlog management
├── planner2.py         # Action planning
├── code_builder.py     # Code generation
├── critic.py           # Code review
├── tool_registry.py    # Tool management
├── executor.py         # Test execution
├── memory.py           # Experience storage
├── requirements.txt    # Dependencies
├── README.md           # This file
├── .env.example        # Environment template
├── backlog.json        # Project backlog
├── tool_registry.json  # Tool registry
├── memory.json         # Experience memory
├── dyn_tools/          # Generated tools
│   └── __init__.py
└── tests/              # Generated tests
    └── __init__.py
```

## 🧪 Testing

Run the test suite:

```bash
pytest -v
```

Run specific tests:

```bash
pytest tests/test_specific_tool.py -v
```

## 🔒 Safety Features

- **Code Review**: All generated code is reviewed by the Critic
- **Pattern Detection**: Blocks dangerous patterns (eval, exec, etc.)
- **Timeout Protection**: Test execution has 30-second timeout
- **Sandboxed Execution**: Tools run in isolated environment
- **Error Handling**: Comprehensive error catching and reporting

## 🧠 Learning & Memory

The system learns from every interaction:

- **Similarity Search**: Finds relevant past experiences
- **Success Tracking**: Remembers what worked and what didn't
- **Pattern Learning**: Improves action selection over time
- **Quality Feedback**: Stores review scores and feedback

## 🚀 Extending the System

### Adding New Tool Types

1. Extend `CodeBuilder.tool_templates`
2. Add pattern matching in `_determine_tool_type()`
3. Implement generation logic in `_generate_implementation()`

### Adding New Actions

1. Add patterns to `Planner2.action_patterns`
2. Implement action handling in `main.py`
3. Update memory storage if needed

### Customizing the Critic

1. Add new safety patterns to `Critic.safety_patterns`
2. Extend quality checks in `_check_quality()`
3. Modify scoring in `_calculate_score()`

## 📊 Monitoring

The system provides comprehensive logging:

```bash
# View system logs
tail -f logs/agent_system.log

# Check memory statistics
python -c "from memory import Memory; m = Memory(); print(m.get_memory_stats())"

# Check tool registry
python -c "from tool_registry import ToolRegistry; tr = ToolRegistry(); print(tr.get_registry_stats())"
```

## 🔮 Future Enhancements

- **Docker Sandboxing**: Containerized tool execution
- **Meta-Critic**: Automated code improvement suggestions
- **FastAPI Wrapper**: HTTP API for external integration
- **Advanced Embeddings**: OpenAI embeddings for better similarity search
- **Multi-threading**: Parallel tool execution
- **Web UI**: Visual interface for tool management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- FAISS for vector similarity search
- Pytest for testing framework
- OpenAI for inspiration on multi-agent systems
- The open-source community for various tools and libraries

---

**Built with ❤️ for autonomous code generation** 