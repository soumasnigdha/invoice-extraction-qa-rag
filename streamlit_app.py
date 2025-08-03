import streamlit as st
import sys
from pathlib import Path
import tempfile
import zipfile
import io
import pandas as pd
from datetime import datetime

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from main import FinancialRAGApp

# Configure Streamlit page
st.set_page_config(
    page_title="Financial RAG Invoice Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize the Financial RAG Application"""
    if 'app' not in st.session_state:
        with st.spinner('Initializing Financial RAG Application...'):
            try:
                st.session_state.app = FinancialRAGApp()
                st.session_state.app_initialized = True
            except Exception as e:
                st.session_state.app_initialized = False
                st.session_state.init_error = str(e)
    
    return st.session_state.get('app_initialized', False)

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory and return paths"""
    temp_dir = Path(tempfile.mkdtemp())
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Save to temp directory
        temp_file_path = temp_dir / uploaded_file.name
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)
    
    return file_paths, temp_dir

def create_download_zip(individual_files, master_file):
    """Create a ZIP file containing all Excel files for download"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add individual invoice files
        for file_path in individual_files:
            if Path(file_path).exists():
                zip_file.write(file_path, f"individual_invoices/{Path(file_path).name}")
        
        # Add master file
        if master_file and Path(master_file).exists():
            zip_file.write(master_file, f"master_data/{Path(master_file).name}")
    
    zip_buffer.seek(0)
    return zip_buffer

def display_processing_results(results):
    """Display processing results in organized format"""
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'error']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(results))
    
    with col2:
        st.metric("Successful", len(successful))
    
    with col3:
        st.metric("Failed", len(failed))
    
    with col4:
        success_rate = (len(successful) / len(results) * 100) if results else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Detailed results
    if successful:
        st.subheader("✅ Successfully Processed")
        success_data = []
        for result in successful:
            success_data.append({
                'Invoice Number': result.get('invoice_number', 'Unknown'),
                'File Name': Path(result.get('file_path', '')).name,
                'Vendor': result.get('extracted_data', {}).get('vendor', {}).get('name', 'Unknown'),
                'Amount': result.get('extracted_data', {}).get('totals', {}).get('grand_total', 0),
                'Currency': result.get('extracted_data', {}).get('invoice_info', {}).get('currency', ''),
                'Validation Errors': len(result.get('validation_results', {}).get('errors', [])),
                'Validation Warnings': len(result.get('validation_results', {}).get('warnings', []))
            })
        
        df_success = pd.DataFrame(success_data)
        st.dataframe(df_success, use_container_width=True)
    
    if failed:
        st.subheader("❌ Processing Errors")
        for result in failed:
            st.error(f"**{Path(result.get('file_path', '')).name}**: {result.get('error', 'Unknown error')}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">📊 Financial RAG Invoice Processor</div>', unsafe_allow_html=True)
    
    # Initialize application
    app_ready = initialize_app()
    
    if not app_ready:
        st.error(f"❌ Failed to initialize application: {st.session_state.get('init_error', 'Unknown error')}")
        st.info("Please check your Ollama installation and ensure llama3 model is available.")
        st.code("ollama pull llama3")
        return
    
    # Sidebar - Application Info
    with st.sidebar:
        st.header("🔧 Application Info")
        
        # Health check
        if st.button("🏥 Health Check"):
            with st.spinner('Checking system health...'):
                health = st.session_state.app.health_check()
                
                if health['status'] == 'healthy':
                    st.success("✅ System is healthy")
                else:
                    st.error("❌ System has issues")
                
                # Show component status
                for component, status in health['components'].items():
                    emoji = "✅" if status['status'] == 'healthy' else "❌"
                    st.write(f"{emoji} **{component.title()}**: {status['status']}")
        
        # Corpus statistics
        if st.button("📊 Corpus Stats"):
            with st.spinner('Getting corpus statistics...'):
                stats = st.session_state.app.get_corpus_stats()
                
                vdb_stats = stats.get('vector_database', {})
                st.metric("Documents in Corpus", vdb_stats.get('unique_documents', 0))
                st.metric("Total Chunks", vdb_stats.get('total_chunks', 0))
                
                excel_stats = stats.get('master_excel', {})
                st.metric("Master Excel Records", excel_stats.get('total_invoices', 0))
        
        # Clear database option
        st.header("🗑️ Maintenance")
        if st.button("Clear Vector Database", type="secondary"):
            if st.session_state.app.clear_vector_database():
                st.success("✅ Vector database cleared")
            else:
                st.error("❌ Failed to clear database")
    
    # Main content
    st.header("📁 Upload Invoices")
    st.info("Upload multiple invoice files (PDF, JPG, PNG, TIFF) for processing using RAG-powered extraction.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose invoice files",
        type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
        accept_multiple_files=True,
        help="Select one or more invoice files to process"
    )
    
    if uploaded_files:
        st.write(f"📋 **{len(uploaded_files)} files selected:**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        # Processing button
        if st.button("🚀 Process Invoices", type="primary", help="Process all uploaded invoices using RAG extraction"):
            with st.spinner(f'Processing {len(uploaded_files)} invoice(s) with RAG...'):
                # Save uploaded files temporarily
                file_paths, temp_dir = save_uploaded_files(uploaded_files)
                
                # Process all files
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file_path in enumerate(file_paths):
                    status_text.text(f'Processing {file_path.name}... ({i+1}/{len(file_paths)})')
                    
                    try:
                        result = st.session_state.app.process_document(file_path)
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'status': 'error',
                            'error': str(e),
                            'file_path': str(file_path)
                        })
                    
                    progress_bar.progress((i + 1) / len(file_paths))
                
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                st.header("📊 Processing Results")
                display_processing_results(results)
                
                # Prepare download files
                successful_results = [r for r in results if r.get('status') == 'success']
                
                if successful_results:
                    st.header("⬇️ Download Processed Files")
                    
                    # Collect all Excel file paths
                    individual_files = []
                    master_file = None
                    
                    for result in successful_results:
                        if 'individual_excel' in result and Path(result['individual_excel']).exists():
                            individual_files.append(result['individual_excel'])
                        if 'master_excel' in result and not master_file:
                            master_file = result['master_excel']
                    
                    # Create download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download individual files as ZIP
                        if individual_files:
                            zip_buffer = create_download_zip(individual_files, master_file)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            st.download_button(
                                label="📦 Download All Excel Files (ZIP)",
                                data=zip_buffer,
                                file_name=f"processed_invoices_{timestamp}.zip",
                                mime="application/zip",
                                help="Download all individual invoice Excel files and master Excel file"
                            )
                    
                    with col2:
                        # Download master Excel file separately
                        if master_file and Path(master_file).exists():
                            with open(master_file, 'rb') as f:
                                st.download_button(
                                    label="📋 Download Master Excel",
                                    data=f.read(),
                                    file_name=Path(master_file).name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Download consolidated master Excel file"
                                )
                
                # Get extraction summary
                summary = st.session_state.app.get_extraction_summary(results)
                
                # Show summary statistics
                with st.expander("📈 Processing Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
                    
                    with col2:
                        st.metric("Validation Errors", summary['validation_errors'])
                    
                    with col3:
                        st.metric("Validation Warnings", summary['validation_warnings'])

    # Query interface
    st.header("🔍 Query Your Invoice Corpus")
    
    query_input = st.text_input(
        "Ask questions about your processed invoices:",
        placeholder="e.g., 'Find all invoices over $10,000' or 'Show me vendors with most transactions'"
    )
    
    if query_input:
        if st.button("🔍 Search"):
            with st.spinner('Searching invoice corpus...'):
                query_result = st.session_state.app.query_documents(query_input)
                
                if 'error' in query_result:
                    st.error(f"Query error: {query_result['error']}")
                else:
                    st.subheader("🎯 Search Results")
                    st.write(f"**Query:** {query_result['query']}")
                    st.write(f"**Found {query_result['total_relevant_chunks']} relevant chunks**")
                    
                    # Display response
                    st.markdown("**Response:**")
                    st.write(query_result['response'])
                    
                    # Show sources
                    if query_result.get('sources'):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(query_result['sources'], 1):
                                st.write(f"**Source {i}:** {source.get('source_file', 'Unknown')}")

if __name__ == "__main__":
    main()
