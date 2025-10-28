# app.py - ENHANCED WITH BATCH UPLOAD, PDF REPORTS, SAMPLE GALLERY & BETTER ERROR HANDLING
import streamlit as st
import os
import sys
from PIL import Image
import tempfile
import base64
from datetime import datetime
import json
import io

# Add src to path to import your inference module
sys.path.append('src')

# Import your MediScanPredictor
from inference import MediScanPredictor

st.set_page_config(
    page_title="MediScan-AI: Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Initialize your model (cache it for performance)
@st.cache_resource
def load_model():
    """Load the MediScan AI model"""
    try:
        predictor = MediScanPredictor(model_path='notebooks/models/resnet50_fold_2_best.pth')
        #predictor = MediScanPredictor(onnx_path='models/mediscan_ai_best.onnx')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==================== PDF REPORT GENERATION ====================
def generate_pdf_report(results, patient_info=None):
    """Generate a PDF report for the analysis results"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart
        
        # Create a bytes buffer for the PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Story to hold PDF elements
        story = []
        
        # Title
        story.append(Paragraph("MediScan-AI Pneumonia Detection Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Patient Information
        if patient_info:
            story.append(Paragraph("Patient Information", heading_style))
            patient_data = [
                ["Name:", patient_info.get('name', 'N/A')],
                ["Age:", patient_info.get('age', 'N/A')],
                ["Gender:", patient_info.get('gender', 'N/A')],
                ["Date:", patient_info.get('date', datetime.now().strftime('%Y-%m-%d'))]
            ]
            patient_table = Table(patient_data, colWidths=[1.5*inch, 3*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Analysis Results
        story.append(Paragraph("Analysis Results", heading_style))
        
        for i, result in enumerate(results):
            story.append(Paragraph(f"Image {i+1}: {result.get('filename', 'Unknown')}", styles['Heading3']))
            
            # Result table
            result_data = [
                ["Prediction:", result['class']],
                ["Confidence:", f"{result['confidence']:.1%}"],
                ["Recommendation:", result['recommendation']],
                ["Model Type:", result['model_type'].upper()],
                ["Analysis Date:", result.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))]
            ]
            
            result_table = Table(result_data, colWidths=[1.5*inch, 4*inch])
            result_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(result_table)
            story.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        story.append(Spacer(1, 0.3*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1  # Center aligned
        )
        story.append(Paragraph("⚠️ IMPORTANT: This is a research prototype for demonstration purposes. Always consult qualified healthcare professionals for medical diagnoses.", disclaimer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except ImportError:
        st.error("PDF generation requires reportlab. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def get_image_download_link(buffer, filename):
    """Create a download link for files"""
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'

# ==================== SAMPLE IMAGE GALLERY ====================
def load_sample_images():
    """Load sample images from samples directory"""
    samples_dir = "samples"
    sample_images = []
    
    if os.path.exists(samples_dir):
        for file in os.listdir(samples_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_images.append(os.path.join(samples_dir, file))
    
    return sample_images

def create_file_uploader_compatible_image(image_path):
    """Convert a local image to a file-like object compatible with st.file_uploader"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Create a file-like object that mimics uploaded file
    from io import BytesIO
    file_like_object = BytesIO(image_data)
    file_like_object.name = os.path.basename(image_path)
    return file_like_object

def main():
    st.title("🫁 MediScan-AI: Pneumonia Detection")
    st.markdown("""
    ### AI-Powered Chest X-Ray Analysis
    
    Upload chest X-ray images to get AI-powered assessments for pneumonia detection.
    """)
    
    # Load model
    with st.spinner("Loading AI model..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("❌ Failed to load model. Please check if model files exist.")
        return
    
    # Initialize session state for Grad-CAM visibility
    if 'show_gradcam' not in st.session_state:
        st.session_state.show_gradcam = {}
    
    # Initialize session state for sample images
    if 'sample_uploaded_files' not in st.session_state:
        st.session_state.sample_uploaded_files = None
    
    # ==================== NEW FEATURE 1: BATCH UPLOAD ====================
    upload_option = st.radio(
        "Choose upload type:",
        ["Single Image", "Multiple Images"],
        horizontal=True
    )
    
    results = []
    
    # Check if we have sample files to process
    if st.session_state.sample_uploaded_files:
        uploaded_files = st.session_state.sample_uploaded_files
        st.session_state.sample_uploaded_files = None  # Reset after use
        st.success(f"✅ Using sample image: {uploaded_files[0].name}")
    else:
        if upload_option == "Single Image":
            uploaded_files = st.file_uploader(
                "Choose a chest X-ray image", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a chest X-ray image in JPG, JPEG, or PNG format"
            )
            if uploaded_files:
                uploaded_files = [uploaded_files]  # Convert to list for consistency
        else:
            uploaded_files = st.file_uploader(
                "Choose chest X-ray images", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload multiple chest X-ray images",
                accept_multiple_files=True
            )
    
    # ==================== PROCESS UPLOADED FILES ====================
    if uploaded_files:
        # Initialize session state for results if not exists
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        # Process each file
        for uploaded_file in uploaded_files:
            try:
                # Display the uploaded image for single mode
                if upload_option == "Single Image":
                    st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Get prediction from your actual model
                with st.spinner(f"🔄 Analyzing {uploaded_file.name}..."):
                    result = predictor.predict_full(temp_path)
                    result['filename'] = uploaded_file.name
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Generate Grad-CAM for PyTorch models (for both single and batch)
                    gradcam_image = None
                    if result['success'] and result['model_type'] == 'pytorch':
                        # For single image, generate Grad-CAM immediately
                        if upload_option == "Single Image":
                            with st.spinner("🔍 Generating attention map..."):
                                gradcam_image = predictor.generate_gradcam(temp_path)
                                result['gradcam'] = gradcam_image
                        # For batch images, we'll generate on-demand when user clicks the button
                        else:
                            result['temp_path'] = temp_path  # Store temp path for later Grad-CAM generation
                    else:
                        # Clean up temp file if not storing for Grad-CAM
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
                
                # Store result
                results.append(result)
                
                # Display results for single image
                if upload_option == "Single Image" and result['success']:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prediction Results")
                        
                        # Color code based on prediction
                        if result['class'] == 'NORMAL':
                            st.success(f"**Predicted Class:** {result['class']}")
                        else:
                            st.error(f"**Predicted Class:** {result['class']}")
                        
                        st.metric("Confidence", f"{result['confidence']:.1%}")
                        
                    with col2:
                        st.subheader("Recommendation")
                        st.info(result['recommendation'])
                    
                    # Show Grad-CAM if available
                    if gradcam_image is not None:
                        st.subheader("🔍 Model Attention Map (Grad-CAM)")
                        st.image(gradcam_image, caption="Red areas show where the model is focusing", use_container_width=True)
                        st.markdown("""
                        **Interpretation:**
                        - 🔴 **Red areas**: Model is paying most attention here
                        - 🟢 **Green areas**: Model is ignoring these regions
                        - The model should focus on lung areas for pneumonia detection
                        """)
                    elif result['model_type'] == 'onnx':
                        st.info("ℹ️ Grad-CAM visualization available only with PyTorch models. Using ONNX for faster inference.")
                    
                    # Show model info
                    st.subheader("Model Information")
                    st.write(f"**Model Type:** {result['model_type'].upper()}")
                    st.write(f"**Processing Time:** Instant")
                
                # Clean up temporary file for single image (already handled above)
                    
            except Exception as e:
                # ==================== NEW FEATURE 4: BETTER ERROR HANDLING ====================
                error_result = {
                    'success': False,
                    'error': f"Error processing {uploaded_file.name}: {str(e)}",
                    'class': 'ERROR',
                    'confidence': 0.0,
                    'recommendation': 'Processing failed - please check the image file',
                    'filename': uploaded_file.name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                results.append(error_result)
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
        # Store results in session state
        st.session_state.analysis_results = results
        
        # ==================== DISPLAY BATCH RESULTS ====================
        if upload_option == "Multiple Images" and results:
            st.subheader("📊 Batch Analysis Results")
            
            # Summary statistics
            successful_results = [r for r in results if r['success']]
            if successful_results:
                normal_count = len([r for r in successful_results if r['class'] == 'NORMAL'])
                pneumonia_count = len([r for r in successful_results if r['class'] == 'PNEUMONIA'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Normal", normal_count)
                with col3:
                    st.metric("Pneumonia", pneumonia_count)
            
            # Detailed results table with Grad-CAM support
            for i, result in enumerate(results):
                with st.expander(f"📄 {result['filename']} - {result['class'] if result['success'] else 'ERROR'}", expanded=False):
                    if result['success']:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Prediction:** {result['class']}")
                            st.write(f"**Confidence:** {result['confidence']:.1%}")
                            
                            # Display uploaded image in batch mode
                            st.image(Image.open(result['temp_path']), caption="Uploaded X-Ray", use_container_width=True)
                            
                        with col2:
                            st.write(f"**Recommendation:** {result['recommendation']}")
                            st.write(f"**Model Type:** {result['model_type'].upper()}")
                            
                            # Grad-CAM functionality for batch images
                            if result['model_type'] == 'pytorch':
                                # Check if Grad-CAM is already generated for this image
                                gradcam_key = f"gradcam_{result['filename']}"
                                
                                if gradcam_key not in st.session_state.show_gradcam:
                                    st.session_state.show_gradcam[gradcam_key] = False
                                
                                # Show/Hide Grad-CAM button
                                if not st.session_state.show_gradcam[gradcam_key]:
                                    if st.button("🔍 Show Grad-CAM", key=f"show_gradcam_{i}"):
                                        st.session_state.show_gradcam[gradcam_key] = True
                                        st.rerun()
                                else:
                                    if st.button("❌ Hide Grad-CAM", key=f"hide_gradcam_{i}"):
                                        st.session_state.show_gradcam[gradcam_key] = False
                                        st.rerun()
                                
                                # Generate and display Grad-CAM when requested
                                if st.session_state.show_gradcam[gradcam_key]:
                                    with st.spinner("Generating attention map..."):
                                        gradcam_image = predictor.generate_gradcam(result['temp_path'])
                                        if gradcam_image is not None:
                                            st.image(gradcam_image, caption="Grad-CAM Visualization", use_container_width=True)
                                            st.markdown("""
                                            **Interpretation:**
                                            - 🔴 **Red areas**: Model is paying most attention here
                                            - 🟢 **Green areas**: Model is ignoring these regions
                                            """)
                                        else:
                                            st.warning("Grad-CAM generation failed for this image.")
                            else:
                                st.info("ℹ️ Grad-CAM visualization available only with PyTorch models.")
                        
                        # Clean up temp file after displaying
                        if 'temp_path' in result and os.path.exists(result['temp_path']):
                            try:
                                os.unlink(result['temp_path'])
                            except:
                                pass
                    else:
                        st.error(f"Error: {result['error']}")
        
        # ==================== PDF REPORT DOWNLOAD ====================
        if results and any(r['success'] for r in results):
            st.subheader("📄 Generate Report")
            
            # Patient information form - MOVED OUTSIDE OF FORM
            st.write("Optional: Add patient information for the report")
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Patient Name", key="patient_name")
                patient_age = st.text_input("Age", key="patient_age")
            with col2:
                patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="patient_gender")
                report_date = st.date_input("Report Date", datetime.now(), key="report_date")
            
            # Generate PDF button - MOVED OUTSIDE OF FORM
            if st.button("📥 Generate PDF Report", key="generate_pdf"):
                patient_info = {
                    'name': patient_name,
                    'age': patient_age,
                    'gender': patient_gender,
                    'date': report_date.strftime('%Y-%m-%d')
                }
                
                pdf_buffer = generate_pdf_report(
                    [r for r in results if r['success']], 
                    patient_info if patient_name else None
                )
                
                if pdf_buffer:
                    st.success("✅ PDF report generated successfully!")
                    
                    # Download button - NOW OUTSIDE OF FORM
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"mediscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )

    # ==================== SIDEBAR ENHANCEMENTS ====================
    with st.sidebar:
        st.header("About MediScan-AI")
        st.markdown("""
        This AI model analyzes chest X-rays to assist in pneumonia detection.
        
        **Model Performance:**
        - Clean Image Accuracy: 100%
        - Provides Confidence Scores
        - Research Prototype
        
        **⚠️ Important Notice:**
        This is a research prototype for demonstration purposes. 
        Always consult qualified healthcare professionals for medical diagnoses.
        """)
        
        # ==================== SAMPLE IMAGE GALLERY ====================
        st.header("🖼️ Sample Image Gallery")
        sample_images = load_sample_images()
        
        if sample_images:
            selected_sample = st.selectbox("Choose a sample image:", sample_images)
            if selected_sample:
                try:
                    sample_img = Image.open(selected_sample)
                    st.image(sample_img, caption=f"Sample: {os.path.basename(selected_sample)}", use_container_width=True)
                    
                    if st.button("🔄 Use This Sample", key="use_sample"):
                        # Create a file-like object from the sample image
                        sample_file = create_file_uploader_compatible_image(selected_sample)
                        # Store in session state to trigger processing
                        st.session_state.sample_uploaded_files = [sample_file]
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
        else:
            st.info("No sample images found. Add images to the 'samples' directory.")
        
        st.header("Features")
        st.markdown("""
        - **Batch Analysis**: Process multiple images at once
        - **PDF Reports**: Download comprehensive analysis reports
        - **Sample Gallery**: Test with pre-loaded sample images
        - **AI-Powered Analysis**: Deep learning model for pneumonia detection
        - **Confidence Scores**: Understand prediction certainty
        - **Grad-CAM Visualization**: See where the model focuses (PyTorch only)
        """)

if __name__ == "__main__":
    main()