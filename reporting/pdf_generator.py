"""
PDF report generation
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
from datetime import datetime
from pathlib import Path

class PDFReportGenerator:
    """Generate comprehensive PDF trading reports"""
    
    def __init__(self, output_dir: str = './reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
    def generate(self, symbol: str, results: dict) -> str:
        """Generate complete PDF report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"trading_report_{symbol.replace('/', '_')}_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        )
        title = Paragraph(f"Crypto Trading Report: {symbol}", title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Summary
        summary_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Symbol:</b> {symbol}<br/>
        <b>Analysis Period:</b> {results.get('period', 'N/A')}<br/>
        """
        summary = Paragraph(summary_text, self.styles['Normal'])
        story.append(summary)
        story.append(Spacer(1, 20))
        
        # Performance Metrics Table
        if 'total_return' in results:
            story.append(Paragraph("<b>Performance Metrics</b>", self.styles['Heading2']))
            story.append(Spacer(1, 12))
            
            metrics_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{results.get('total_return', 0):.2%}"],
                ['Sharpe Ratio', f"{results.get('sharpe_ratio', 0):.2f}"],
                ['Sortino Ratio', f"{results.get('sortino_ratio', 0):.2f}"],
                ['Max Drawdown', f"{results.get('max_drawdown', 0):.2%}"],
                ['Win Rate', f"{results.get('win_rate', 0):.2%}"],
                ['Profit Factor', f"{results.get('profit_factor', 0):.2f}"],
                ['Calmar Ratio', f"{results.get('calmar_ratio', 0):.2f}"]
            ]
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 20))
            
        # Equity Curve Chart
        if 'equity_curve' in results:
            story.append(Paragraph("<b>Equity Curve</b>", self.styles['Heading2']))
            equity_chart = self._create_equity_chart(results['equity_curve'])
            story.append(equity_chart)
            story.append(Spacer(1, 20))
            
        # Build PDF
        doc.build(story)
        return str(filepath)
        
    def _create_equity_chart(self, equity_df):
        """Create equity curve chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(equity_df.index, equity_df['portfolio_value'], linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Equity Curve')
        ax.grid(True, alpha=0.3)
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Create image
        img = Image(buf, width=6*inch, height=3.6*inch)
        return img
