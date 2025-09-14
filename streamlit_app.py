"""
Streamlit Web Interface for Multi-Agent Market Research System
"""

import streamlit as st
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Configure page
st.set_page_config(
    page_title="Multi-Agent Market Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import system modules
try:
    from main import get_orchestrator
    from config.settings import SUPPORTED_INDUSTRIES, USE_CASE_CATEGORIES, REPORTS_DIR, DATASETS_DIR
    from utils.logger import get_logger, log_user_action
    from utils.validation import validate_research_input
except ImportError as e:
    st.error(f"Failed to import system modules: {e}")
    st.stop()

# Initialize logger
logger = get_logger(__name__)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2e7d32);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    try:
        st.session_state.orchestrator = get_orchestrator()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        st.stop()

if 'execution_results' not in st.session_state:
    st.session_state.execution_results = None

if 'research_history' not in st.session_state:
    st.session_state.research_history = []

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-Agent Market Research System</h1>
        <p>AI-Powered Industry Analysis & Use Case Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Research", "📊 Dashboard", "📁 Results", "⚙️ System"])
    
    with tab1:
        research_interface()
    
    with tab2:
        dashboard_interface()
    
    with tab3:
        results_interface()
    
    with tab4:
        system_interface()

def setup_sidebar():
    """Setup sidebar with system information and controls"""
    
    st.sidebar.markdown("### 🚀 System Status")
    
    try:
        status = st.session_state.orchestrator.get_system_status()
        
        if status["system_status"] == "operational":
            st.sidebar.success("✅ System Operational")
        else:
            st.sidebar.error("❌ System Issues")
        
        st.sidebar.metric("Total Executions", status.get("total_executions", 0))
        st.sidebar.metric("Success Rate", f"{status.get('successful_executions', 0)}/{status.get('total_executions', 0)}")
        
        if status.get("last_execution"):
            st.sidebar.text(f"Last Execution: {status['last_execution'][:19]}")
        
    except Exception as e:
        st.sidebar.error(f"Status Error: {e}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Actions")
    
    if st.sidebar.button("🔄 Refresh System"):
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.research_history = []
        st.session_state.execution_results = None
        st.sidebar.success("History cleared!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    This system uses multiple AI agents to:
    - Research companies and industries
    - Identify AI/ML opportunities
    - Generate implementation roadmaps
    - Find relevant datasets
    """)

def research_interface():
    """Main research interface"""
    
    st.markdown("## 🔍 Market Research & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input form
        with st.form("research_form"):
            st.markdown("### Research Parameters")
            
            company_name = st.text_input(
                "Company/Organization Name *",
                placeholder="e.g., Microsoft, Tesla, Local Manufacturing Inc.",
                help="Enter the name of the company or organization you want to research"
            )
            
            industry = st.selectbox(
                "Industry Sector *",
                options=[""] + SUPPORTED_INDUSTRIES,
                help="Select the primary industry sector"
            )
            
            description = st.text_area(
                "Additional Context (Optional)",
                placeholder="Provide any additional context about the company, specific challenges, or research focus...",
                height=100,
                help="Optional: Add specific context to help guide the research"
            )
            
            depth = st.select_slider(
                "Research Depth",
                options=["quick", "standard", "deep"],
                value="standard",
                help="Quick: 4 searches, Standard: 6 searches, Deep: 8+ searches"
            )
            
            submitted = st.form_submit_button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            handle_research_submission(company_name, industry, description, depth)
    
    with col2:
        st.markdown("### 📊 Research Progress")
        
        if st.session_state.execution_results:
            display_progress_summary()
        else:
            st.info("Submit research parameters to begin analysis")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        history_count = len(st.session_state.research_history)
        st.metric("Research Sessions", history_count)
        
        if history_count > 0:
            recent_session = st.session_state.research_history[-1]
            st.text(f"Last: {recent_session.get('company_name', 'N/A')}")

def handle_research_submission(company_name: str, industry: str, description: str, depth: str):
    """Handle research form submission"""
    
    # Validate inputs
    is_valid, errors = validate_research_input(company_name, industry or "Technology", description)
    
    if not is_valid:
        st.error("Please fix the following errors:")
        for error in errors:
            st.error(f"• {error}")
        return
    
    # Log user action
    log_user_action("research_started", details={
        "company": company_name,
        "industry": industry,
        "depth": depth
    })
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Executing multi-agent research pipeline..."):
        try:
            # Execute research
            status_text.text("Starting research agents...")
            progress_bar.progress(10)
            
            results = st.session_state.orchestrator.execute_research_pipeline(
                company_name=company_name,
                industry=industry,
                description=description,
                depth=depth
            )
            
            progress_bar.progress(100)
            status_text.text("Research completed!")
            
            if results.get("success"):
                st.session_state.execution_results = results
                
                # Add to history
                st.session_state.research_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "company_name": company_name,
                    "industry": industry,
                    "depth": depth,
                    "execution_id": results["execution_id"],
                    "success": True
                })
                
                st.success("Research completed successfully! Check the Results tab for detailed reports.")
                
            else:
                st.error(f"Research failed: {results.get('error', 'Unknown error')}")
                if 'errors' in results:
                    for error in results['errors']:
                        st.error(f"• {error}")
        
        except Exception as e:
            logger.error(f"Research execution failed: {str(e)}")
            st.error(f"System error: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

def display_progress_summary():
    """Display progress summary of current research"""
    
    results = st.session_state.execution_results
    if not results:
        return
    
    stages = results.get("stages", {})
    
    # Overall status
    if results.get("success"):
        st.success("✅ Research Completed")
    else:
        st.error("❌ Research Failed")
    
    # Stage progress
    st.markdown("#### Stage Progress")
    
    stage_info = [
        ("research", "Industry Research", stages.get("research", {}).get("success", False)),
        ("datasets", "Dataset Discovery", stages.get("datasets", {}).get("success", False)),
        ("use_cases", "Use Case Generation", stages.get("use_cases", {}).get("success", False)),
        ("report", "Report Generation", stages.get("report", {}).get("success", False))
    ]
    
    for stage_key, stage_name, stage_success in stage_info:
        if stage_success:
            st.success(f"✅ {stage_name}")
        elif stage_key in stages:
            st.error(f"❌ {stage_name}")
        else:
            st.info(f"⏳ {stage_name}")

def dashboard_interface():
    """Dashboard showing research analytics and insights"""
    
    st.markdown("## 📊 Research Dashboard")
    
    if not st.session_state.execution_results:
        st.info("No research results available. Please run a research analysis first.")
        return
    
    results = st.session_state.execution_results
    stages = results.get("stages", {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Target", results.get("input", {}).get("company_name", "N/A"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        use_cases = stages.get("use_cases", {}).get("use_cases", [])
        st.metric("Use Cases Found", len(use_cases))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        datasets = stages.get("datasets", {}).get("datasets", {})
        total_datasets = sum(len(d) for d in datasets.values())
        st.metric("Datasets Found", total_datasets)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        search_results = stages.get("research", {}).get("search_results", [])
        st.metric("Sources Analyzed", len(search_results))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Use cases breakdown
    if use_cases:
        st.markdown("### 🎯 Use Cases Overview")
        
        # Priority distribution
        priorities = [uc.get("priority", "Medium") for uc in use_cases]
        priority_counts = {p: priorities.count(p) for p in ["High", "Medium", "Low"]}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Priority Distribution")
            for priority, count in priority_counts.items():
                st.metric(f"{priority} Priority", count)
        
        with col2:
            st.markdown("#### Implementation Roadmap")
            roadmap = stages.get("use_cases", {}).get("implementation_roadmap", {})
            
            if roadmap:
                phase1 = roadmap.get("phase_1_quick_wins", [])
                phase2 = roadmap.get("phase_2_core_implementations", [])
                phase3 = roadmap.get("phase_3_advanced_initiatives", [])
                
                st.metric("Phase 1 (Quick Wins)", len(phase1))
                st.metric("Phase 2 (Core)", len(phase2))
                st.metric("Phase 3 (Advanced)", len(phase3))
    
    # Dataset breakdown
    if total_datasets > 0:
        st.markdown("### 📊 Dataset Sources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kaggle_count = len(datasets.get("kaggle", []))
            st.metric("Kaggle", kaggle_count)
        
        with col2:
            hf_count = len(datasets.get("huggingface", []))
            st.metric("HuggingFace", hf_count)
        
        with col3:
            github_count = len(datasets.get("github", []))
            st.metric("GitHub", github_count)

def results_interface():
    """Interface for viewing and downloading results"""
    
    st.markdown("## 📁 Research Results")
    
    if not st.session_state.execution_results:
        st.info("No research results available. Please run a research analysis first.")
        return
    
    results = st.session_state.execution_results
    stages = results.get("stages", {})
    
    # Tabs for different result types
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Reports", "🎯 Use Cases", "📊 Datasets", "🔍 Raw Data"])
    
    with tab1:
        display_reports(results, stages)
    
    with tab2:
        display_use_cases(stages.get("use_cases", {}))
    
    with tab3:
        display_datasets(stages.get("datasets", {}))
    
    with tab4:
        display_raw_data(results)

def display_reports(results: Dict[str, Any], stages: Dict[str, Any]):
    """Display generated reports"""
    
    st.markdown("### 📄 Generated Reports")
    
    report_info = stages.get("report", {})
    
    if report_info.get("success"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ Main Research Report Generated")
            report_path = report_info.get("report_path", "")
            if report_path and Path(report_path).exists():
                st.download_button(
                    "📥 Download Research Report",
                    data=Path(report_path).read_text(encoding='utf-8'),
                    file_name=f"research_report_{results['execution_id']}.md",
                    mime="text/markdown"
                )
        
        with col2:
            st.success("✅ Dataset Report Generated")
            dataset_path = report_info.get("dataset_report_path", "")
            if dataset_path and Path(dataset_path).exists():
                st.download_button(
                    "📥 Download Dataset Report",
                    data=Path(dataset_path).read_text(encoding='utf-8'),
                    file_name=f"datasets_report_{results['execution_id']}.md",
                    mime="text/markdown"
                )
    
    # Research findings summary
    research_stage = stages.get("research", {})
    if research_stage.get("success") and research_stage.get("comprehensive_analysis"):
        st.markdown("### 📋 Research Summary")
        st.markdown(research_stage["comprehensive_analysis"][:1000] + "...")
        
        with st.expander("View Full Analysis"):
            st.markdown(research_stage["comprehensive_analysis"])

def display_use_cases(use_cases_stage: Dict[str, Any]):
    """Display generated use cases"""
    
    use_cases = use_cases_stage.get("use_cases", [])
    
    if not use_cases:
        st.info("No use cases generated.")
        return
    
    st.markdown(f"### 🎯 AI/ML Use Cases ({len(use_cases)} found)")
    
    # Priority filter
    priorities = list(set([uc.get("priority", "Medium") for uc in use_cases]))
    selected_priority = st.selectbox("Filter by Priority", ["All"] + priorities)
    
    filtered_cases = use_cases
    if selected_priority != "All":
        filtered_cases = [uc for uc in use_cases if uc.get("priority") == selected_priority]
    
    # Display use cases
    for i, use_case in enumerate(filtered_cases, 1):
        with st.expander(f"{i}. {use_case.get('title', 'Unnamed Use Case')} [{use_case.get('priority', 'Medium')} Priority]"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Category:**")
                st.write(use_case.get("category", "Not specified"))
                
                st.markdown("**Problem Statement:**")
                st.write(use_case.get("problem_statement", "Not specified"))
                
                st.markdown("**Proposed Solution:**")
                st.write(use_case.get("solution", "Not specified"))
            
            with col2:
                st.markdown("**Expected Benefits:**")
                st.write(use_case.get("benefits", "Not specified"))
                
                st.markdown("**Implementation Details:**")
                st.write(f"**Complexity:** {use_case.get('complexity', 'Medium')}")
                st.write(f"**Timeline:** {use_case.get('timeline', 'Not specified')}")
                st.write(f"**Investment Level:** {use_case.get('investment_level', 'Medium')}")
                
                if use_case.get("priority_score"):
                    st.write(f"**Priority Score:** {use_case['priority_score']:.1f}/100")

def display_datasets(datasets_stage: Dict[str, Any]):
    """Display discovered datasets"""
    
    datasets = datasets_stage.get("datasets", {})
    
    if not datasets:
        st.info("No datasets discovered.")
        return
    
    total_datasets = sum(len(d) for d in datasets.values())
    st.markdown(f"### 📊 Discovered Datasets ({total_datasets} found)")
    
    # Platform tabs
    if datasets.get("kaggle"):
        st.markdown("#### Kaggle Datasets")
        display_dataset_list(datasets["kaggle"], "kaggle")
    
    if datasets.get("huggingface"):
        st.markdown("#### HuggingFace Datasets")
        display_dataset_list(datasets["huggingface"], "huggingface")
    
    if datasets.get("github"):
        st.markdown("#### GitHub Repositories")
        display_dataset_list(datasets["github"], "github")

def display_dataset_list(dataset_list: List[Dict[str, Any]], platform: str):
    """Display a list of datasets for a specific platform"""
    
    for i, dataset in enumerate(dataset_list[:10], 1):  # Limit to top 10
        with st.expander(f"{i}. {dataset.get('title', 'Unnamed Dataset')}"):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**URL:** [{dataset.get('url', '#')}]({dataset.get('url', '#')})")
                st.markdown(f"**Author:** {dataset.get('author', 'Unknown')}")
                st.markdown(f"**Description:** {dataset.get('description', 'No description available')}")
                
                # Platform-specific metrics
                if platform == "kaggle":
                    downloads = dataset.get("download_count", 0)
                    votes = dataset.get("vote_count", 0)
                    usability = dataset.get("usability_rating", 0)
                    st.write(f"Downloads: {downloads:,} | Votes: {votes} | Usability: {usability}/10")
                
                elif platform == "huggingface":
                    downloads = dataset.get("downloads", 0)
                    likes = dataset.get("likes", 0)
                    st.write(f"Downloads: {downloads:,} | Likes: {likes}")
                
                elif platform == "github":
                    stars = dataset.get("stars", 0)
                    forks = dataset.get("forks", 0)
                    language = dataset.get("language", "N/A")
                    st.write(f"Stars: {stars:,} | Forks: {forks} | Language: {language}")
            
            with col2:
                if dataset.get("relevance_score"):
                    score = dataset["relevance_score"]
                    st.metric("Relevance", f"{score:.1f}/100")

def display_raw_data(results: Dict[str, Any]):
    """Display raw execution data"""
    
    st.markdown("### 🔍 Raw Execution Data")
    
    # Execution metadata
    st.markdown("#### Execution Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Execution ID:** {results.get('execution_id', 'N/A')}")
        st.write(f"**Timestamp:** {results.get('timestamp', 'N/A')}")
    
    with col2:
        st.write(f"**Success:** {results.get('success', False)}")
        input_data = results.get('input', {})
        st.write(f"**Depth:** {input_data.get('depth', 'N/A')}")
    
    with col3:
        execution_time = results.get('execution_time', 0)
        st.write(f"**Duration:** {execution_time:.2f}s")
    
    # Raw JSON data
    if st.checkbox("Show Raw JSON Data"):
        st.json(results)

def system_interface():
    """System monitoring and configuration interface"""
    
    st.markdown("## ⚙️ System Information")
    
    try:
        orchestrator = st.session_state.orchestrator
        system_status = orchestrator.get_system_status()
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### System Status")
            if system_status["system_status"] == "operational":
                st.success("✅ Operational")
            else:
                st.error("❌ Issues Detected")
            
            st.metric("Total Executions", system_status.get("total_executions", 0))
            st.metric("Successful", system_status.get("successful_executions", 0))
        
        with col2:
            st.markdown("### Agent Status")
            research_status = system_status.get("research_agent_status", {})
            st.write(f"**Research Agent:** {research_status.get('status', 'Unknown')}")
            st.metric("Agent Executions", research_status.get("execution_count", 0))
            st.metric("Success Rate", f"{research_status.get('success_rate', 0):.1f}%")
        
        with col3:
            st.markdown("### Recent Activity")
            execution_history = orchestrator.get_execution_history(5)
            
            if execution_history:
                for exec_info in execution_history[-3:]:
                    timestamp = exec_info.get("timestamp", "").split("T")[0]
                    success = exec_info.get("success", False)
                    status_icon = "✅" if success else "❌"
                    st.write(f"{status_icon} {timestamp}")
            else:
                st.info("No recent activity")
        
        # Configuration
        st.markdown("### 🔧 Configuration")
        
        with st.expander("View System Configuration"):
            config_info = {
                "Supported Industries": len(SUPPORTED_INDUSTRIES),
                "Use Case Categories": len(USE_CASE_CATEGORIES),
                "Reports Directory": str(REPORTS_DIR),
                "Datasets Directory": str(DATASETS_DIR)
            }
            
            for key, value in config_info.items():
                st.write(f"**{key}:** {value}")
        
        # File browser
        st.markdown("### 📂 Output Files")
        
        if st.button("🔄 Refresh File List"):
            st.rerun()
        
        # List recent reports
        if REPORTS_DIR.exists():
            report_files = list(REPORTS_DIR.glob("*.md"))
            if report_files:
                st.markdown("#### Recent Reports")
                for report_file in sorted(report_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]:
                    file_size = report_file.stat().st_size
                    st.write(f"📄 {report_file.name} ({file_size:,} bytes)")
            else:
                st.info("No report files found")
        
    except Exception as e:
        st.error(f"Failed to load system information: {e}")

if __name__ == "__main__":
    main()