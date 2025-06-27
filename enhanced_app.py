"""Enhanced Streamlit UI for Document Q&A Agent with intelligence features."""

import streamlit as st
from pathlib import Path
import os
from src.qa_engine import QAEngine
from src.enhanced_qa_engine import EnhancedQAEngine
from src.utils.config import get_config
from src.utils.logger import logger


# Page configuration
st.set_page_config(
    page_title="Enhanced Document Q&A Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "engine_type" not in st.session_state:
    st.session_state.engine_type = "enhanced"

if "qa_engine" not in st.session_state:
    try:
        if st.session_state.engine_type == "enhanced":
            st.session_state.qa_engine = EnhancedQAEngine()
        else:
            st.session_state.qa_engine = QAEngine()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.error(f"Failed to initialize: {str(e)}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()


def switch_engine(engine_type: str):
    """Switch between standard and enhanced QA engines."""
    if engine_type != st.session_state.engine_type:
        st.session_state.engine_type = engine_type
        try:
            if engine_type == "enhanced":
                st.session_state.qa_engine = EnhancedQAEngine()
            else:
                st.session_state.qa_engine = QAEngine()
            st.success(f"Switched to {engine_type} engine")
        except Exception as e:
            st.error(f"Failed to switch engine: {str(e)}")


def format_validation_result(validation):
    """Format validation result for display."""
    if not validation:
        return ""
    
    result = f"\n\n**Answer Validation:**\n"
    result += f"- **Confidence Level:** {validation.confidence_level}\n"
    result += f"- **Complete Answer:** {'Yes' if validation.is_complete else 'No'}\n"
    
    if validation.missing_information:
        result += f"- **Missing Information:** {', '.join(validation.missing_information)}\n"
    
    if validation.contradictions:
        result += f"- **Contradictions Found:** {', '.join(validation.contradictions)}\n"
    
    if validation.suggestions:
        result += f"- **Suggestions:** {', '.join(validation.suggestions)}\n"
    
    return result


def format_query_analysis(analysis):
    """Format query analysis for display."""
    if not analysis:
        return ""
    
    result = f"\n**Query Analysis:**\n"
    result += f"- **Type:** {analysis.query_type.value}\n"
    result += f"- **Complexity:** {analysis.complexity_score:.2f}\n"
    result += f"- **Multi-hop:** {'Yes' if analysis.multi_hop else 'No'}\n"
    
    if analysis.entities:
        result += f"- **Key Entities:** {', '.join(analysis.entities)}\n"
    
    return result


def main():
    """Main application function."""
    
    # Header
    st.title("🧠 Enhanced Document Q&A Agent")
    st.markdown("Upload documents and ask questions using advanced RAG-powered AI")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Engine selection
        engine_type = st.radio(
            "Select Engine",
            ["standard", "enhanced"],
            index=1 if st.session_state.engine_type == "enhanced" else 0,
            help="Enhanced engine includes query classification, multi-stage retrieval, and answer validation"
        )
        
        if engine_type != st.session_state.engine_type:
            switch_engine(engine_type)
        
        st.divider()
        
        st.header("📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=["pdf"],
            help="Upload PDF files to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("Processing document..."):
                    try:
                        # Process the uploaded file
                        result = st.session_state.qa_engine.load_uploaded_file(
                            uploaded_file.read(),
                            uploaded_file.name
                        )
                        
                        if result["success"]:
                            st.session_state.uploaded_files.add(uploaded_file.name)
                            # Get the appropriate message based on action
                            action = result.get("action", "added")
                            if action == "unchanged":
                                st.info(f"Document {uploaded_file.name} already exists with same content")
                            else:
                                chunks_indexed = result.get("chunks_indexed", "Unknown")
                                st.success(
                                    f"✅ Loaded {uploaded_file.name}\n\n"
                                    f"- Action: {action}\n"
                                    f"- Chunks indexed: {chunks_indexed}"
                                )
                        else:
                            st.error(f"Failed to load document: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
        
        # Document list and management
        st.divider()
        st.header("📄 Indexed Documents")
        
        if st.session_state.initialized and st.session_state.engine_type == "enhanced":
            documents = st.session_state.qa_engine.list_documents()
            
            if documents:
                for doc in documents:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(doc["filename"])
                    with col2:
                        st.text(f"{doc['chunk_count']} chunks")
                    with col3:
                        if st.button("🗑️", key=f"delete_{doc['filename']}", help=f"Delete {doc['filename']}"):
                            result = st.session_state.qa_engine.delete_document(doc["filename"])
                            if result["success"]:
                                st.success(f"Deleted {doc['filename']}")
                                # Remove from uploaded files set
                                st.session_state.uploaded_files.discard(doc["filename"])
                                st.rerun()
                            else:
                                st.error(f"Failed to delete: {result.get('error', 'Unknown error')}")
            else:
                st.info("No documents indexed yet")
        
        # System stats
        st.divider()
        st.header("📊 System Status")
        
        if st.session_state.initialized:
            stats = st.session_state.qa_engine.get_stats()
            
            if st.session_state.engine_type == "enhanced":
                # Show document management stats
                doc_stats = stats.get("document_management", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Total Documents",
                        doc_stats.get("total_documents", 0)
                    )
                with col2:
                    integrity = "✅ Healthy" if doc_stats.get("integrity_check", False) else "⚠️ Issues"
                    st.metric(
                        "System Integrity",
                        integrity
                    )
            else:
                st.metric(
                    "Documents in Vector Store",
                    stats["vector_store"].get("document_count", 0)
                )
            
            # Show enhanced features if using enhanced engine
            if st.session_state.engine_type == "enhanced" and "enhanced_features" in stats:
                st.subheader("Enhanced Features")
                features = stats["enhanced_features"]
                for feature, enabled in features.items():
                    st.checkbox(
                        feature.replace("_", " ").title(),
                        value=enabled,
                        disabled=True,
                        key=f"feature_{feature}"
                    )
            
            with st.expander("Configuration"):
                config = stats["config"]
                st.json(config)
        
        # Clear documents button
        st.divider()
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.qa_engine.clear_documents()
            st.session_state.uploaded_files.clear()
            st.session_state.messages = []
            st.success("All documents cleared!")
            st.rerun()
    
    # Main chat interface
    if not st.session_state.initialized:
        st.error("System not initialized. Please check your configuration.")
        return
    
    # Chat settings
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        use_rag = st.checkbox("Use RAG", value=True, help="Enable document retrieval")
    with col3:
        if st.session_state.engine_type == "enhanced":
            validate_answer = st.checkbox("Validate Answer", value=True, help="Enable answer validation")
        else:
            validate_answer = False
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show additional information for enhanced engine
            if message["role"] == "assistant" and st.session_state.engine_type == "enhanced":
                if "query_analysis" in message:
                    with st.expander("🔍 Query Analysis"):
                        st.markdown(format_query_analysis(message["query_analysis"]))
                
                if "validation" in message and message.get("validation"):
                    with st.expander("✅ Answer Validation"):
                        st.markdown(format_validation_result(message["validation"]))
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📄 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            metadata = source.get("metadata", {})
                            st.markdown(
                                f"**Source {i}:** {metadata.get('filename', 'Unknown')} "
                                f"(chunk {metadata.get('chunk_index', '?')})"
                            )
                            if st.session_state.engine_type == "enhanced":
                                # Show relevance score if available
                                score = source.get('adjusted_score', source.get('distance', 'N/A'))
                                st.caption(f"Relevance score: {score}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            sources = []
            query_analysis = None
            validation_result = None
            
            try:
                # Get response with streaming
                if st.session_state.engine_type == "enhanced":
                    result = st.session_state.qa_engine.answer_question(
                        prompt,
                        use_rag=use_rag,
                        stream=True,
                        validate=validate_answer and use_rag
                    )
                else:
                    result = st.session_state.qa_engine.answer_question(
                        prompt,
                        use_rag=use_rag,
                        stream=True
                    )
                
                if result["success"]:
                    # Stream the response
                    for chunk in result["stream"]:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    sources = result.get("sources", [])
                    
                    # Get additional info for enhanced engine
                    if st.session_state.engine_type == "enhanced":
                        query_analysis = result.get("query_analysis")
                        
                        # Get validation if not streaming
                        if validate_answer and use_rag:
                            # Re-run without streaming to get validation
                            validation_result_full = st.session_state.qa_engine.answer_question(
                                prompt,
                                use_rag=use_rag,
                                stream=False,
                                validate=True
                            )
                            if validation_result_full["success"]:
                                validation_result = validation_result_full.get("validation")
                    
                    # Show query analysis
                    if query_analysis:
                        with st.expander("🔍 Query Analysis"):
                            st.markdown(format_query_analysis(query_analysis))
                    
                    # Show validation
                    if validation_result:
                        with st.expander("✅ Answer Validation"):
                            st.markdown(format_validation_result(validation_result))
                    
                    # Show sources
                    if sources:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(sources, 1):
                                metadata = source.get("metadata", {})
                                st.markdown(
                                    f"**Source {i}:** {metadata.get('filename', 'Unknown')} "
                                    f"(chunk {metadata.get('chunk_index', '?')})"
                                )
                                if st.session_state.engine_type == "enhanced":
                                    score = source.get('adjusted_score', source.get('distance', 'N/A'))
                                    st.caption(f"Relevance score: {score}")
                else:
                    full_response = f"Error: {result.get('error', 'Unknown error')}"
                    message_placeholder.error(full_response)
            
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.error(full_response)
        
        # Add assistant message to history
        message_data = {
            "role": "assistant",
            "content": full_response,
            "sources": sources
        }
        
        if st.session_state.engine_type == "enhanced":
            if query_analysis:
                message_data["query_analysis"] = query_analysis
            if validation_result:
                message_data["validation"] = validation_result
        
        st.session_state.messages.append(message_data)


if __name__ == "__main__":
    # Check for API key
    try:
        config = get_config()
        main()
    except ValueError as e:
        st.error(
            "⚠️ **Configuration Error**\n\n"
            f"{str(e)}\n\n"
            "Please create a `.env` file with your OpenAI API key:\n"
            "```\n"
            "OPENAI_API_KEY=your_api_key_here\n"
            "```"
        )
