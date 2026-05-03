"""
Report Generator for Drowsiness Detection System
Generates PDF and Excel reports from logging data
"""

import os
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from io import BytesIO


class ReportGenerator:
    """Generate PDF and Excel reports for drowsiness detection data"""
    
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._add_custom_styles()
    
    def _add_custom_styles(self):
        """Add custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6
        ))
    
    def _create_chart(self, data: List[Dict], chart_type: str) -> BytesIO:
        """Create matplotlib chart and return as BytesIO"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        if chart_type == 'hourly_trend':
            # Hourly drowsiness trend
            hours = [d['hour'] for d in data]
            counts = [d['count'] for d in data]
            ax.plot(hours, counts, marker='o', linewidth=2, color='#3b82f6')
            ax.set_xlabel('Giờ trong ngày', fontsize=12)
            ax.set_ylabel('Số lượt ngủ gật', fontsize=12)
            ax.set_title('Xu hướng ngủ gật theo giờ', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        elif chart_type == 'camera_comparison':
            # Camera comparison bar chart
            cameras = [d['camera_name'] for d in data]
            counts = [d['total_events'] for d in data]
            ax.bar(cameras, counts, color='#3b82f6', alpha=0.7)
            ax.set_xlabel('Phòng học', fontsize=12)
            ax.set_ylabel('Số sự kiện', fontsize=12)
            ax.set_title('So sánh số lượng ngủ gật giữa các phòng', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to BytesIO
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        return img_buffer
    
    def generate_pdf_report(
        self,
        period: str,
        camera_stats: List[Dict[str, Any]],
        summary: Dict[str, Any],
        events: List[Dict[str, Any]],
        camera_ids: Optional[List[str]] = None
    ) -> str:
        """
        Generate PDF report
        
        Args:
            period: Time period (today, week, month, custom)
            camera_stats: Statistics for each camera
            summary: Summary statistics
            events: Detailed events list
            camera_ids: Optional list of camera IDs to include
            
        Returns:
            Path to generated PDF file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'drowsiness_report_{period}_{timestamp}.pdf'
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )
        
        story = []
        
        # Title
        title_text = 'BÁO CÁO GIÁM SÁT NGỦ GẬT HỌC SINH'
        title = Paragraph(title_text, self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Period info
        period_text = f"<b>Khoảng thời gian:</b> {self._format_period(period, summary)}"
        story.append(Paragraph(period_text, self.styles['CustomBody']))
        
        report_time = f"<b>Thời gian tạo báo cáo:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        story.append(Paragraph(report_time, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary section
        story.append(Paragraph('TỔNG QUAN', self.styles['CustomHeading']))
        
        summary_data = [
            ['Chỉ số', 'Giá trị'],
            ['Tổng số phòng giám sát', str(summary.get('total_cameras', 0))],
            ['Tổng học sinh ngủ gật (duy nhất)', str(summary.get('total_drowsy_students_unique', 0))],
            ['Tổng số sự kiện', str(summary.get('total_events', 0))],
            ['Tổng thời gian ngủ gật', summary.get('total_duration_display', '0s')],
            ['Đang ngủ gật (hiện tại)', str(summary.get('currently_drowsy_all_cameras', 0))]
        ]
        
        summary_table = Table(summary_data, colWidths=[3.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Camera statistics
        if camera_stats:
            story.append(Paragraph('THỐNG KÊ THEO PHÒNG HỌC', self.styles['CustomHeading']))
            
            camera_data = [['Phòng học', 'Học sinh ngủ gật', 'Số sự kiện', 'Thời gian', 'Đang ngủ']]
            for stat in camera_stats:
                camera_data.append([
                    stat.get('camera_name', 'N/A'),
                    str(stat.get('total_drowsy_students', 0)),
                    str(stat.get('total_events', 0)),
                    stat.get('total_duration_display', '0s'),
                    str(stat.get('currently_drowsy', 0))
                ])
            
            camera_table = Table(camera_data, colWidths=[2*inch, 1.2*inch, 1*inch, 1*inch, 0.8*inch])
            camera_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
            ]))
            
            story.append(camera_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Detailed events (limited to first 20)
        if events:
            story.append(PageBreak())
            story.append(Paragraph('CHI TIẾT SỰ KIỆN NGỦ GẬT', self.styles['CustomHeading']))
            
            display_events = events[:20]  # Limit for PDF
            
            event_data = [['Phòng', 'Học sinh', 'Bắt đầu', 'Kết thúc', 'Thời lượng']]
            for event in display_events:
                event_data.append([
                    event.get('camera_name', 'N/A')[:15],
                    f"#{event.get('student_id', '?')}",
                    event.get('start_time', 'N/A')[-8:],  # Time only
                    event.get('end_time', 'N/A')[-8:] if not event.get('is_active') else 'Đang ngủ',
                    event.get('duration_display', '0s')
                ])
            
            event_table = Table(event_data, colWidths=[1.5*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.3*inch])
            event_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f4f6')])
            ]))
            
            story.append(event_table)
            
            if len(events) > 20:
                note = f"<i>Chỉ hiển thị 20/{len(events)} sự kiện. Xem file Excel để có danh sách đầy đủ.</i>"
                story.append(Spacer(1, 0.1*inch))
                story.append(Paragraph(note, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        
        print(f"✅ PDF report generated: {filepath}")
        return filepath
    
    def generate_excel_report(
        self,
        period: str,
        camera_stats: List[Dict[str, Any]],
        summary: Dict[str, Any],
        events: List[Dict[str, Any]]
    ) -> str:
        """
        Generate Excel report with multiple sheets
        
        Returns:
            Path to generated Excel file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'drowsiness_report_{period}_{timestamp}.xlsx'
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_df = pd.DataFrame([{
                'Khoảng thời gian': self._format_period(period, summary),
                'Tổng số phòng': summary.get('total_cameras', 0),
                'Tổng học sinh ngủ gật': summary.get('total_drowsy_students_unique', 0),
                'Tổng số sự kiện': summary.get('total_events', 0),
                'Tổng thời gian (giây)': summary.get('total_duration_seconds', 0),
                'Tổng thời gian (hiển thị)': summary.get('total_duration_display', '0s'),
                'Đang ngủ gật': summary.get('currently_drowsy_all_cameras', 0),
                'Thời gian tạo báo cáo': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }])
            summary_df.to_excel(writer, sheet_name='Tổng quan', index=False)
            
            # Sheet 2: Camera Statistics
            if camera_stats:
                camera_df = pd.DataFrame(camera_stats)
                # Rename columns to Vietnamese
                column_mapping = {
                    'camera_id': 'Mã camera',
                    'camera_name': 'Tên phòng',
                    'total_drowsy_students': 'Số học sinh ngủ gật',
                    'currently_drowsy': 'Đang ngủ gật',
                    'total_events': 'Số sự kiện',
                    'total_duration_seconds': 'Thời gian (giây)',
                    'total_duration_display': 'Thời gian (hiển thị)',
                    'average_duration_seconds': 'Trung bình (giây)',
                    'longest_duration_seconds': 'Lâu nhất (giây)'
                }
                camera_df = camera_df.rename(columns=column_mapping)
                camera_df.to_excel(writer, sheet_name='Thống kê phòng', index=False)
            
            # Sheet 3: Detailed Events
            if events:
                events_df = pd.DataFrame(events)
                # Rename columns
                event_column_mapping = {
                    'camera_id': 'Mã camera',
                    'camera_name': 'Tên phòng',
                    'student_id': 'Mã học sinh',
                    'start_time': 'Thời gian bắt đầu',
                    'end_time': 'Thời gian kết thúc',
                    'duration_seconds': 'Thời lượng (giây)',
                    'duration_display': 'Thời lượng (hiển thị)',
                    'is_active': 'Đang diễn ra'
                }
                events_df = events_df.rename(columns=event_column_mapping)
                events_df.to_excel(writer, sheet_name='Chi tiết sự kiện', index=False)
        
        print(f"✅ Excel report generated: {filepath}")
        return filepath
    
    def _format_period(self, period: str, summary: Dict[str, Any]) -> str:
        """Format period string for display"""
        if period == 'today':
            return 'Hôm nay'
        elif period == 'week':
            return 'Tuần này'
        elif period == 'month':
            return 'Tháng này'
        else:
            start = summary.get('period_start', '')
            end = summary.get('period_end', '')
            if start and end:
                return f"{start[:10]} → {end[:10]}"
            return period
    
    def cleanup_old_reports(self, days: int = 30):
        """Delete reports older than specified days"""
        import time
        cutoff_time = time.time() - (days * 86400)
        
        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath):
                if os.path.getmtime(filepath) < cutoff_time:
                    os.remove(filepath)
                    print(f"🗑️ Deleted old report: {filename}")


# Global instance
_report_generator = None

def get_report_generator(output_dir='reports') -> ReportGenerator:
    """Get or create global report generator instance"""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator(output_dir)
    return _report_generator
