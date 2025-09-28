#!/usr/bin/env python3
"""
Báo cáo Tổng hợp Kết quả Training 1000 Epochs
==========================================

Script này sẽ phân tích và báo cáo tất cả kết quả training từ các models
YOLOv5, YOLOv8, và YOLOv11 sau quá trình training 1000 epochs.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

class TrainingResultsAnalyzer:
    """Phân tích kết quả training tổng hợp"""
    
    def __init__(self):
        self.root_dir = Path("..").resolve()
        self.runs_dir = self.root_dir / "runs" / "pose-train"
        self.backup_dir = self.root_dir / "model_backups_1000ep"
        self.results_dir = self.root_dir / "training_results_1000ep"
        
        self.analysis_results = {}
        
    def analyze_training_runs(self):
        """Phân tích các training runs"""
        print("🔍 Analyzing training runs...")
        print(f"📂 Looking in: {self.runs_dir}")
        
        if not self.runs_dir.exists():
            print("❌ No training runs directory found")
            return {}
            
        runs_analysis = {}
        
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir():
                print(f"\n📁 Checking: {run_dir.name}")
                
                # Phân tích weights
                weights_dir = run_dir / "weights"
                weights_info = {}
                
                if weights_dir.exists():
                    for weight_file in weights_dir.glob("*.pt"):
                        size_mb = weight_file.stat().st_size / 1024 / 1024
                        weights_info[weight_file.name] = {
                            "size_mb": size_mb,
                            "created": datetime.fromtimestamp(weight_file.stat().st_mtime).isoformat()
                        }
                        print(f"   💾 Found: {weight_file.name} ({size_mb:.1f}MB)")
                
                # Phân tích results.csv nếu có
                results_csv = run_dir / "results.csv"
                training_metrics = {}
                
                if results_csv.exists():
                    try:
                        df = pd.read_csv(results_csv)
                        if len(df) > 0:
                            # Tìm các cột có sẵn
                            available_cols = df.columns.tolist()
                            print(f"   📊 CSV columns: {len(available_cols)} columns")
                            
                            training_metrics = {
                                "epochs_completed": len(df),
                                "csv_columns": available_cols[:10]  # Lấy 10 cột đầu tiên
                            }
                            
                            # Thêm metrics nếu có
                            metrics_cols = [col for col in df.columns if 'mAP' in col or 'loss' in col]
                            if metrics_cols:
                                last_row = df.iloc[-1]
                                for col in metrics_cols[:5]:  # Lấy 5 metrics đầu tiên
                                    training_metrics[f"final_{col}"] = float(last_row[col]) if pd.notna(last_row[col]) else None
                                    
                                # Best values
                                for col in metrics_cols[:5]:
                                    if 'mAP' in col:
                                        training_metrics[f"best_{col}"] = float(df[col].max()) if pd.notna(df[col].max()) else None
                            
                            print(f"   📈 Epochs completed: {training_metrics['epochs_completed']}")
                            
                    except Exception as e:
                        print(f"   ❌ Error reading results.csv: {e}")
                        training_metrics = {"error": str(e)}
                
                # Kiểm tra log files
                log_files = list(run_dir.glob("*.log")) + list(run_dir.glob("*.txt"))
                log_info = {}
                for log_file in log_files:
                    size_kb = log_file.stat().st_size / 1024
                    log_info[log_file.name] = {"size_kb": size_kb}
                
                runs_analysis[run_dir.name] = {
                    "weights": weights_info,
                    "metrics": training_metrics,
                    "logs": log_info,
                    "status": "completed" if "best.pt" in weights_info else "incomplete"
                }
        
        self.analysis_results["training_runs"] = runs_analysis
        return runs_analysis
    
    def analyze_model_backups(self):
        """Phân tích model backups"""
        print(f"\n🔍 Analyzing model backups...")
        print(f"📂 Looking in: {self.backup_dir}")
        
        if not self.backup_dir.exists():
            print("❌ No backup directory found")
            return {}
            
        backups_analysis = {}
        
        for backup_file in self.backup_dir.glob("*.pt"):
            size_mb = backup_file.stat().st_size / 1024 / 1024
            created = datetime.fromtimestamp(backup_file.stat().st_mtime)
            
            backups_analysis[backup_file.name] = {
                "size_mb": size_mb,
                "created": created.isoformat(),
                "model_type": self._extract_model_type(backup_file.name)
            }
            print(f"   💾 {backup_file.name}: {size_mb:.1f}MB")
        
        self.analysis_results["model_backups"] = backups_analysis
        return backups_analysis
    
    def analyze_training_reports(self):
        """Phân tích training reports"""
        print(f"\n🔍 Analyzing training reports...")
        print(f"📂 Looking in: {self.results_dir}")
        
        if not self.results_dir.exists():
            print("❌ No results directory found")
            return {}
            
        reports_analysis = {}
        
        # Tìm tất cả JSON files
        for report_file in self.results_dir.glob("*.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    reports_analysis[report_file.name] = report_data
                    print(f"   📄 {report_file.name}: loaded")
            except Exception as e:
                print(f"   ❌ Error reading {report_file.name}: {e}")
        
        # Tìm log files
        for log_file in self.results_dir.glob("*.log"):
            size_kb = log_file.stat().st_size / 1024
            print(f"   📝 {log_file.name}: {size_kb:.1f}KB")
        
        self.analysis_results["training_reports"] = reports_analysis
        return reports_analysis
    
    def check_existing_models(self):
        """Kiểm tra models hiện có"""
        print(f"\n🔍 Checking existing models in root directory...")
        
        model_files = {}
        
        # Tìm trong thư mục gốc
        for pattern in ["*.pt", "yolo*.pt"]:
            for model_file in self.root_dir.glob(pattern):
                if model_file.is_file():
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    created = datetime.fromtimestamp(model_file.stat().st_mtime)
                    
                    model_files[model_file.name] = {
                        "size_mb": size_mb,
                        "created": created.isoformat(),
                        "model_type": self._extract_model_type(model_file.name),
                        "path": str(model_file)
                    }
                    print(f"   🤖 {model_file.name}: {size_mb:.1f}MB")
        
        self.analysis_results["existing_models"] = model_files
        return model_files
    
    def check_training_logs(self):
        """Kiểm tra training logs"""
        print(f"\n🔍 Checking training logs...")
        
        log_files = {}
        
        # Tìm trong tools directory
        tools_dir = Path(".")
        for log_file in tools_dir.glob("*.log"):
            size_kb = log_file.stat().st_size / 1024
            modified = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            log_files[log_file.name] = {
                "size_kb": size_kb,
                "modified": modified.isoformat(),
                "path": str(log_file)
            }
            print(f"   📝 {log_file.name}: {size_kb:.1f}KB (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
        
        self.analysis_results["training_logs"] = log_files
        return log_files
    
    def _extract_model_type(self, filename):
        """Trích xuất loại model từ tên file"""
        filename_lower = filename.lower()
        if "yolov11" in filename_lower or "v11" in filename_lower:
            return "YOLOv11"
        elif "yolov8" in filename_lower or "v8" in filename_lower:
            return "YOLOv8"
        elif "yolov5" in filename_lower or "v5" in filename_lower:
            return "YOLOv5"
        else:
            return "Unknown"
    
    def generate_comprehensive_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n📋 Generating comprehensive report...")
        print("="*80)
        
        # Phân tích tất cả dữ liệu
        runs_analysis = self.analyze_training_runs()
        backups_analysis = self.analyze_model_backups()
        reports_analysis = self.analyze_training_reports()
        existing_models = self.check_existing_models()
        training_logs = self.check_training_logs()
        
        # Tạo báo cáo tổng hợp
        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_summary": {
                "total_training_runs": len(runs_analysis),
                "completed_runs": sum(1 for r in runs_analysis.values() if r.get("status") == "completed"),
                "total_model_backups": len(backups_analysis),
                "total_training_reports": len(reports_analysis),
                "existing_models_count": len(existing_models),
                "training_logs_count": len(training_logs)
            },
            "detailed_analysis": self.analysis_results
        }
        
        # Lưu báo cáo
        report_filename = f"comprehensive_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = Path(".") / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # In báo cáo tóm tắt
        self.print_summary_report(report)
        
        return report_path
    
    def print_summary_report(self, report):
        """In báo cáo tóm tắt"""
        print("\n" + "="*80)
        print("🎉 BÁO CÁO TỔNG HỢP TRAINING 1000 EPOCHS")
        print("="*80)
        
        summary = report["analysis_summary"]
        print(f"\n📊 TỔNG QUAN:")
        print(f"   🏃 Tổng số training runs: {summary['total_training_runs']}")
        print(f"   ✅ Runs hoàn thành: {summary['completed_runs']}")
        print(f"   💾 Model backups: {summary['total_model_backups']}")
        print(f"   📄 Training reports: {summary['total_training_reports']}")
        print(f"   🤖 Models hiện có: {summary['existing_models_count']}")
        print(f"   📝 Training logs: {summary['training_logs_count']}")
        
        # Chi tiết training runs
        runs = self.analysis_results.get("training_runs", {})
        if runs:
            print(f"\n🏃 CHI TIẾT TRAINING RUNS:")
            for run_name, run_data in runs.items():
                model_type = self._extract_model_type(run_name)
                metrics = run_data.get("metrics", {})
                status = run_data.get("status", "unknown")
                weights = run_data.get("weights", {})
                
                print(f"\n   📁 {model_type} - {run_name}:")
                print(f"      📊 Status: {status}")
                
                if "epochs_completed" in metrics:
                    print(f"      🎯 Epochs completed: {metrics['epochs_completed']}")
                
                if "error" in metrics:
                    print(f"      ❌ Error: {metrics['error']}")
                
                # Show available metrics
                metric_keys = [k for k in metrics.keys() if k.startswith(('final_', 'best_'))]
                if metric_keys:
                    print(f"      📈 Available metrics: {len(metric_keys)}")
                    for key in metric_keys[:3]:  # Show first 3 metrics
                        value = metrics[key]
                        if value is not None:
                            print(f"         {key}: {value:.4f}")
                
                if weights:
                    print(f"      💾 Model files:")
                    for weight_name, weight_info in weights.items():
                        size_mb = weight_info["size_mb"]
                        print(f"         {weight_name}: {size_mb:.1f}MB")
        else:
            print(f"\n❌ Không tìm thấy training runs nào!")
        
        # Models hiện có
        existing = self.analysis_results.get("existing_models", {})
        if existing:
            print(f"\n🤖 MODELS HIỆN CÓ:")
            for model_name, model_info in existing.items():
                model_type = model_info["model_type"]
                size_mb = model_info["size_mb"]
                print(f"   📦 {model_type}: {model_name} ({size_mb:.1f}MB)")
        
        # Training logs
        logs = self.analysis_results.get("training_logs", {})
        if logs:
            print(f"\n📝 TRAINING LOGS:")
            for log_name, log_info in logs.items():
                size_kb = log_info["size_kb"]
                modified = log_info["modified"][:16]  # Just date and time
                print(f"   📄 {log_name}: {size_kb:.1f}KB (modified: {modified})")
        
        # Model backups
        backups = self.analysis_results.get("model_backups", {})
        if backups:
            print(f"\n💾 MODEL BACKUPS:")
            for backup_name, backup_info in backups.items():
                model_type = backup_info["model_type"]
                size_mb = backup_info["size_mb"]
                print(f"   📦 {model_type}: {backup_name} ({size_mb:.1f}MB)")
        
        # Training status assessment
        print(f"\n🎯 ĐÁNH GIÁ TÌNH TRẠNG TRAINING:")
        
        if summary["completed_runs"] == 0:
            print("   🔴 Chưa có training runs nào hoàn thành")
            print("   💡 Khuyến nghị: Chạy script train_all_models_1000_epochs.py")
        elif summary["completed_runs"] < 3:
            print(f"   🟡 Có {summary['completed_runs']}/3 models đã training xong")
            print("   💡 Khuyến nghị: Tiếp tục training các models còn lại")
        else:
            print("   🟢 Tất cả 3 models đã training xong!")
            print("   💡 Sẵn sàng cho testing và deployment")
        
        print(f"\n⏱️  THỜI GIAN TRAINING DỰ KIẾN:")
        print(f"   🔥 1000 epochs per model: ~8-12 giờ/model")
        print(f"   🚀 Tổng thời gian cho 3 models: ~24-36 giờ")
        
        print("\n" + "="*80)
        print("✨ TRAINING ANALYSIS COMPLETED!")
        print("="*80)

def main():
    """Main function"""
    print("🎯 BÁO CÁO TỔNG HỢP TRAINING 1000 EPOCHS")
    print("="*50)
    
    analyzer = TrainingResultsAnalyzer()
    report_path = analyzer.generate_comprehensive_report()
    
    print(f"\n📄 Comprehensive report saved: {report_path}")
    print("\n🔗 Để tiếp tục training:")
    print("   python train_all_models_1000_epochs.py")
    print("\n🔗 Để test models:")
    print("   python test_all_models.py")

if __name__ == "__main__":
    main()