document.addEventListener('DOMContentLoaded', function() {
    const downloadButton = document.getElementById('downloadPdfButton');
    const contentToPrint = document.getElementById('printableArea');

    if (!downloadButton || !contentToPrint) return;

    downloadButton.addEventListener('click', async function(e) {
        e.preventDefault();

        // Wait for all images to load
        const images = contentToPrint.getElementsByTagName('img');
        await Promise.all([...images].map(img => 
            img.complete ? Promise.resolve() : new Promise(resolve => img.onload = resolve)
        ));

        // Backup original styles
        const originalFilter = contentToPrint.style.backdropFilter;
        const originalBg = contentToPrint.style.background;
        const originalAnimation = contentToPrint.style.animation;

        // Apply PDF-safe styles
        contentToPrint.style.backdropFilter = "none";
        contentToPrint.style.background = "#ffffff";
        contentToPrint.style.animation = "none";

        // Optionally, add a class for PDF-safe overrides
        contentToPrint.classList.add("pdf-safe");

        // Capture as canvas
        html2canvas(contentToPrint, {
            scale: 2,
            useCORS: true,
            backgroundColor: "#ffffff"
        }).then(canvas => {
            // Restore original styles
            contentToPrint.style.backdropFilter = originalFilter;
            contentToPrint.style.background = originalBg;
            contentToPrint.style.animation = originalAnimation;
            contentToPrint.classList.remove("pdf-safe");

            // Generate PDF
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF('p', 'mm', 'a4');

            const imgData = canvas.toDataURL('image/jpeg', 1.0);
            const imgProps = pdf.getImageProperties(imgData);

            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

            pdf.addImage(imgData, 'JPEG', 0, 0, pdfWidth, pdfHeight);

            // Get patient ID for filename
            const idElem = contentToPrint.querySelector('span'); // Your patient_id span
            const patientId = idElem ? idElem.textContent.trim() : 'Unknown';

            pdf.save(`NeuroScan_Report_${patientId}.pdf`);
        }).catch(err => {
            console.error("Error generating PDF:", err);
            alert("Failed to generate PDF. See console for details.");
        });
    });
});
