"""
Attendance logs viewing window for FaceAttend application
Implements Phase 4: Data Management functionality including log viewing, filtering, and export
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import pandas as pd
import csv
from pathlib import Path

from src.utils.logger import get_module_logger
from src.storage.attendance_logger import AttendanceLogger
from src.storage.face_storage import FaceStorage

class LogsWindow:
    """Attendance logs viewing and management window"""
    
    def __init__(self, parent: tk.Tk = None):
        """
        Initialize the logs window
        
        Args:
            parent: Parent window
        """
        self.logger = get_module_logger("LogsWindow")
        self.parent = parent
        self.window = None
        self.is_window_open = False
        
        # Initialize data managers
        self.attendance_logger = AttendanceLogger()
        self.face_storage = FaceStorage()
        
        # Data storage
        self.current_records = []
        self.filtered_records = []
        
        # UI components
        self.tree = None
        self.stats_text = None
        self.date_from_var = None
        self.date_to_var = None
        self.user_filter_var = None
        self.search_var = None
        
        self.logger.info("LogsWindow initialized")
    
    def show_window(self):
        """Show the logs window"""
        try:
            if self.window is None or not self.is_window_open:
                self._create_window()
            else:
                self.window.lift()
                self.window.focus_force()
                
        except Exception as e:
            self.logger.error(f"Failed to show logs window: {str(e)}")
            raise
    
    def _create_window(self):
        """Create the logs window UI"""
        try:
            # Create window
            self.window = tk.Toplevel(self.parent) if self.parent else tk.Tk()
            self.window.title("FaceAttend - Attendance Logs")
            self.window.geometry("1200x800")
            self.window.resizable(True, True)
            
            # Set window properties
            self.is_window_open = True
            self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
            
            # Create main UI
            self._setup_ui()
            
            # Load initial data
            self._load_recent_logs()
            
            self.logger.info("Logs window created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating logs window: {str(e)}")
            raise
    
    def _setup_ui(self):
        """Set up the user interface"""
        try:
            # Create main container
            main_frame = ttk.Frame(self.window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Title
            title_label = ttk.Label(
                main_frame, 
                text="📊 Attendance Logs Viewer", 
                font=("Arial", 18, "bold")
            )
            title_label.pack(pady=(0, 15))
            
            # Create sections
            self._create_filter_section(main_frame)
            self._create_logs_section(main_frame)
            self._create_stats_section(main_frame)
            self._create_controls_section(main_frame)
            
        except Exception as e:
            self.logger.error(f"Error setting up UI: {str(e)}")
            raise
    
    def _create_filter_section(self, parent):
        """Create the filtering section"""
        # Filter frame
        filter_frame = ttk.LabelFrame(parent, text="🔍 Filters & Search", padding=15)
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date range section
        date_frame = ttk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date from
        ttk.Label(date_frame, text="From Date:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.date_from_var = tk.StringVar()
        date_from_entry = ttk.Entry(date_frame, textvariable=self.date_from_var, width=12)
        date_from_entry.grid(row=0, column=1, padx=(0, 15))
        date_from_entry.bind('<KeyRelease>', self._on_filter_change)
        
        # Date to
        ttk.Label(date_frame, text="To Date:").grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
        self.date_to_var = tk.StringVar()
        date_to_entry = ttk.Entry(date_frame, textvariable=self.date_to_var, width=12)
        date_to_entry.grid(row=0, column=3, padx=(0, 15))
        date_to_entry.bind('<KeyRelease>', self._on_filter_change)
        
        # Quick date buttons
        quick_date_frame = ttk.Frame(date_frame)
        quick_date_frame.grid(row=0, column=4, padx=(15, 0))
        
        ttk.Button(quick_date_frame, text="Today", command=self._set_today, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_date_frame, text="This Week", command=self._set_this_week, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_date_frame, text="This Month", command=self._set_this_month, width=10).pack(side=tk.LEFT)
        
        # User and search filters
        search_frame = ttk.Frame(filter_frame)
        search_frame.pack(fill=tk.X)
        
        # User filter
        ttk.Label(search_frame, text="User Filter:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.user_filter_var = tk.StringVar()
        user_filter_combo = ttk.Combobox(search_frame, textvariable=self.user_filter_var, width=20, state="readonly")
        user_filter_combo.grid(row=0, column=1, padx=(0, 15))
        user_filter_combo.bind('<<ComboboxSelected>>', self._on_filter_change)
        self.user_filter_combo = user_filter_combo
        
        # Search
        ttk.Label(search_frame, text="Search:").grid(row=0, column=2, padx=(0, 5), sticky=tk.W)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.grid(row=0, column=3, padx=(0, 15))
        search_entry.bind('<KeyRelease>', self._on_filter_change)
        
        # Control buttons
        button_frame = ttk.Frame(search_frame)
        button_frame.grid(row=0, column=4, padx=(15, 0))
        
        ttk.Button(button_frame, text="🔄 Refresh", command=self._refresh_data, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🗑️ Clear", command=self._clear_filters, width=10).pack(side=tk.LEFT)
        
        # Set default dates (last 30 days)
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        self.date_from_var.set(start_date.strftime('%Y-%m-%d'))
        self.date_to_var.set(end_date.strftime('%Y-%m-%d'))
    
    def _create_logs_section(self, parent):
        """Create the logs viewing section"""
        # Logs frame
        logs_frame = ttk.LabelFrame(parent, text="📋 Attendance Records", padding=10)
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(logs_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        columns = ('Date', 'Time', 'User ID', 'Name', 'Confidence')
        self.tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Define column headings and widths
        self.tree.heading('Date', text='Date')
        self.tree.heading('Time', text='Time')
        self.tree.heading('User ID', text='User ID')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Confidence', text='Confidence')
        
        self.tree.column('Date', width=100)
        self.tree.column('Time', width=80)
        self.tree.column('User ID', width=120)
        self.tree.column('Name', width=150)
        self.tree.column('Confidence', width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click for record details
        self.tree.bind('<Double-1>', self._show_record_details)
    
    def _create_stats_section(self, parent):
        """Create the statistics section"""
        # Stats frame
        stats_frame = ttk.LabelFrame(parent, text="📈 Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stats text widget
        self.stats_text = tk.Text(stats_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.pack(fill=tk.X)
        
        # Configure text styling
        self.stats_text.tag_configure("header", font=("Arial", 10, "bold"))
        self.stats_text.tag_configure("value", font=("Arial", 10))
    
    def _create_controls_section(self, parent):
        """Create the control buttons section"""
        # Controls frame
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X)
        
        # Export section
        export_frame = ttk.LabelFrame(controls_frame, text="📤 Export", padding=10)
        export_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        export_button_frame = ttk.Frame(export_frame)
        export_button_frame.pack()
        
        ttk.Button(export_button_frame, text="📋 Export Current View", command=self._export_current_view, width=20).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_button_frame, text="📊 Export All Data", command=self._export_all_data, width=20).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(export_button_frame, text="📆 Export Date Range", command=self._export_date_range, width=20).pack(side=tk.LEFT)
        
        # Admin section
        admin_frame = ttk.LabelFrame(controls_frame, text="🔧 Admin", padding=10)
        admin_frame.pack(side=tk.RIGHT, padx=(5, 0))
        
        admin_button_frame = ttk.Frame(admin_frame)
        admin_button_frame.pack()
        
        ttk.Button(admin_button_frame, text="🗑️ Cleanup Old Logs", command=self._cleanup_old_logs, width=18).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(admin_button_frame, text="📁 Open Logs Folder", command=self._open_logs_folder, width=18).pack(side=tk.LEFT)
    
    def _load_recent_logs(self):
        """Load recent attendance logs"""
        try:
            # Get date range from filters
            start_date = self._parse_date(self.date_from_var.get())
            end_date = self._parse_date(self.date_to_var.get())
            
            if start_date and end_date:
                self.current_records = self.attendance_logger.get_attendance_range(start_date, end_date)
            else:
                # Default to last 30 days
                end_date = date.today()
                start_date = end_date - timedelta(days=30)
                self.current_records = self.attendance_logger.get_attendance_range(start_date, end_date)
            
            # Update user filter dropdown
            self._update_user_filter()
            
            # Apply filters and update display
            self._apply_filters()
            self._update_statistics()
            
            self.logger.info(f"Loaded {len(self.current_records)} attendance records")
            
        except Exception as e:
            self.logger.error(f"Error loading attendance logs: {str(e)}")
            messagebox.showerror("Error", f"Failed to load attendance logs: {str(e)}")
    
    def _update_user_filter(self):
        """Update the user filter dropdown with available users"""
        try:
            # Get unique users from current records
            users = set()
            for record in self.current_records:
                users.add(f"{record['name']} ({record['user_id']})")
            
            # Update combobox
            user_list = ['All Users'] + sorted(list(users))
            self.user_filter_combo['values'] = user_list
            
            # Set default selection if not already set
            if not self.user_filter_var.get():
                self.user_filter_var.set('All Users')
                
        except Exception as e:
            self.logger.error(f"Error updating user filter: {str(e)}")
    
    def _apply_filters(self):
        """Apply current filters to the records"""
        try:
            filtered_records = self.current_records.copy()
            
            # Apply user filter
            user_filter = self.user_filter_var.get()
            if user_filter and user_filter != 'All Users':
                # Extract user_id from the filter string
                user_id = user_filter.split('(')[-1].split(')')[0]
                filtered_records = [r for r in filtered_records if r['user_id'] == user_id]
            
            # Apply search filter
            search_term = self.search_var.get().lower()
            if search_term:
                filtered_records = [
                    r for r in filtered_records 
                    if search_term in r['name'].lower() or 
                       search_term in r['user_id'].lower() or
                       search_term in r['date'].lower() or
                       search_term in r['time'].lower()
                ]
            
            self.filtered_records = filtered_records
            self._update_tree_view()
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
    
    def _update_tree_view(self):
        """Update the tree view with filtered records"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Add filtered records (most recent first)
            for record in reversed(self.filtered_records):
                values = (
                    record['date'],
                    record['time'],
                    record['user_id'],
                    record['name'],
                    f"{float(record['confidence']):.1f}"
                )
                self.tree.insert('', 'end', values=values)
            
            # Update status
            total_count = len(self.current_records)
            filtered_count = len(self.filtered_records)
            status = f"Showing {filtered_count} of {total_count} records"
            
            # Update window title with count
            self.window.title(f"FaceAttend - Attendance Logs ({status})")
            
        except Exception as e:
            self.logger.error(f"Error updating tree view: {str(e)}")
    
    def _update_statistics(self):
        """Update the statistics display"""
        try:
            # Get statistics from attendance logger
            start_date = self._parse_date(self.date_from_var.get())
            end_date = self._parse_date(self.date_to_var.get())
            
            if start_date and end_date:
                stats = self.attendance_logger.get_attendance_statistics(start_date, end_date)
            else:
                stats = self.attendance_logger.get_attendance_statistics()
            
            # Format statistics text
            stats_text = f"""📊 Period: {stats.get('date_range', 'N/A')} | """
            stats_text += f"📝 Total Entries: {stats.get('total_entries', 0)} | "
            stats_text += f"👥 Unique Users: {stats.get('unique_users', 0)} | "
            stats_text += f"📈 Daily Average: {stats.get('avg_daily_attendance', 0):.1f} entries/day"
            
            if self.filtered_records != self.current_records:
                stats_text += f" | 🔍 Filtered: {len(self.filtered_records)} shown"
            
            # Update stats display
            self.stats_text.configure(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_text)
            self.stats_text.configure(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error updating statistics: {str(e)}")
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string to date object"""
        try:
            if not date_str:
                return None
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return None
    
    def _set_today(self):
        """Set date range to today"""
        today = date.today()
        self.date_from_var.set(today.strftime('%Y-%m-%d'))
        self.date_to_var.set(today.strftime('%Y-%m-%d'))
        self._refresh_data()
    
    def _set_this_week(self):
        """Set date range to this week"""
        today = date.today()
        start_of_week = today - timedelta(days=today.weekday())
        self.date_from_var.set(start_of_week.strftime('%Y-%m-%d'))
        self.date_to_var.set(today.strftime('%Y-%m-%d'))
        self._refresh_data()
    
    def _set_this_month(self):
        """Set date range to this month"""
        today = date.today()
        start_of_month = today.replace(day=1)
        self.date_from_var.set(start_of_month.strftime('%Y-%m-%d'))
        self.date_to_var.set(today.strftime('%Y-%m-%d'))
        self._refresh_data()
    
    def _clear_filters(self):
        """Clear all filters"""
        self.user_filter_var.set('All Users')
        self.search_var.set('')
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        self.date_from_var.set(start_date.strftime('%Y-%m-%d'))
        self.date_to_var.set(end_date.strftime('%Y-%m-%d'))
        self._refresh_data()
    
    def _refresh_data(self):
        """Refresh the data from source"""
        self._load_recent_logs()
    
    def _on_filter_change(self, event=None):
        """Handle filter change events"""
        # If date changed, reload data
        if event and event.widget in [self.date_from_var, self.date_to_var]:
            self._load_recent_logs()
        else:
            # Just apply filters
            self._apply_filters()
            self._update_statistics()
    
    def _show_record_details(self, event):
        """Show detailed information for selected record"""
        try:
            selection = self.tree.selection()
            if not selection:
                return
            
            # Get record data
            item = self.tree.item(selection[0])
            values = item['values']
            
            if len(values) >= 5:
                details = f"""
📋 Attendance Record Details

📅 Date: {values[0]}
🕐 Time: {values[1]}
🆔 User ID: {values[2]}
👤 Name: {values[3]}
🎯 Confidence: {values[4]}

📊 Additional Information:
• Recognition Quality: {'Excellent' if float(values[4]) < 100 else 'Good' if float(values[4]) < 150 else 'Fair'}
• Timestamp: {values[0]} {values[1]}
                """
                
                messagebox.showinfo("Record Details", details)
                
        except Exception as e:
            self.logger.error(f"Error showing record details: {str(e)}")
    
    def _export_current_view(self):
        """Export currently filtered records"""
        try:
            if not self.filtered_records:
                messagebox.showwarning("No Data", "No records to export")
                return
            
            # Get export file path
            filename = f"attendance_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=filename
            )
            
            if file_path:
                # Export filtered records
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp', 'user_id', 'name', 'confidence', 'date', 'time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.filtered_records)
                
                messagebox.showinfo("Export Complete", f"Exported {len(self.filtered_records)} records to:\n{file_path}")
                self.logger.info(f"Exported {len(self.filtered_records)} filtered records to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting current view: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _export_all_data(self):
        """Export all attendance data"""
        try:
            # Get all records (no date filter)
            all_records = self.attendance_logger.get_attendance_range(
                date(2020, 1, 1), 
                date.today()
            )
            
            if not all_records:
                messagebox.showwarning("No Data", "No attendance records found")
                return
            
            # Get export file path
            filename = f"attendance_all_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=filename
            )
            
            if file_path:
                # Export all records
                with open(file_path, 'w', newline='') as csvfile:
                    fieldnames = ['timestamp', 'user_id', 'name', 'confidence', 'date', 'time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_records)
                
                messagebox.showinfo("Export Complete", f"Exported {len(all_records)} total records to:\n{file_path}")
                self.logger.info(f"Exported {len(all_records)} total records to {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error exporting all data: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _export_date_range(self):
        """Export data for a custom date range"""
        try:
            # Create date range dialog
            dialog = DateRangeDialog(self.window)
            if dialog.result:
                start_date, end_date = dialog.result
                
                # Get records for date range
                records = self.attendance_logger.get_attendance_range(start_date, end_date)
                
                if not records:
                    messagebox.showwarning("No Data", f"No records found for {start_date} to {end_date}")
                    return
                
                # Get export file path
                filename = f"attendance_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    initialname=filename
                )
                
                if file_path:
                    # Export records
                    with open(file_path, 'w', newline='') as csvfile:
                        fieldnames = ['timestamp', 'user_id', 'name', 'confidence', 'date', 'time']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(records)
                    
                    messagebox.showinfo("Export Complete", f"Exported {len(records)} records to:\n{file_path}")
                    self.logger.info(f"Exported {len(records)} records for range {start_date}-{end_date} to {file_path}")
                    
        except Exception as e:
            self.logger.error(f"Error exporting date range: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            # Ask for confirmation
            result = messagebox.askyesno(
                "Cleanup Old Logs",
                "This will delete log files older than 90 days.\n\nDo you want to continue?"
            )
            
            if result:
                deleted_count = self.attendance_logger.cleanup_old_logs(90)
                messagebox.showinfo("Cleanup Complete", f"Deleted {deleted_count} old log files")
                self.logger.info(f"Cleaned up {deleted_count} old log files")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up logs: {str(e)}")
            messagebox.showerror("Cleanup Error", f"Failed to cleanup logs: {str(e)}")
    
    def _open_logs_folder(self):
        """Open the logs folder in system file manager"""
        try:
            import subprocess
            import platform
            
            logs_path = self.attendance_logger.base_directory
            
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(logs_path)])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", str(logs_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(logs_path)])
                
            self.logger.info(f"Opened logs folder: {logs_path}")
            
        except Exception as e:
            self.logger.error(f"Error opening logs folder: {str(e)}")
            messagebox.showerror("Error", f"Failed to open logs folder: {str(e)}")
    
    def _on_window_close(self):
        """Handle window close event"""
        try:
            self.is_window_open = False
            if self.window:
                self.window.destroy()
                self.window = None
            
            self.logger.info("Logs window closed")
            
        except Exception as e:
            self.logger.error(f"Error closing logs window: {str(e)}")


class DateRangeDialog:
    """Dialog for selecting date range"""
    
    def __init__(self, parent):
        self.result = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Select Date Range")
        self.dialog.geometry("300x200")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.transient(parent)
        
        # Create UI
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Start date
        ttk.Label(main_frame, text="Start Date (YYYY-MM-DD):").pack(anchor=tk.W, pady=(0, 5))
        self.start_var = tk.StringVar()
        start_entry = ttk.Entry(main_frame, textvariable=self.start_var, width=30)
        start_entry.pack(fill=tk.X, pady=(0, 15))
        
        # End date
        ttk.Label(main_frame, text="End Date (YYYY-MM-DD):").pack(anchor=tk.W, pady=(0, 5))
        self.end_var = tk.StringVar()
        end_entry = ttk.Entry(main_frame, textvariable=self.end_var, width=30)
        end_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Set default values
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        self.start_var.set(start_date.strftime('%Y-%m-%d'))
        self.end_var.set(end_date.strftime('%Y-%m-%d'))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="OK", command=self._ok).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Focus and wait
        start_entry.focus()
        self.dialog.wait_window()
    
    def _ok(self):
        try:
            start_date = datetime.strptime(self.start_var.get(), '%Y-%m-%d').date()
            end_date = datetime.strptime(self.end_var.get(), '%Y-%m-%d').date()
            
            if start_date > end_date:
                messagebox.showerror("Invalid Range", "Start date must be before end date")
                return
            
            self.result = (start_date, end_date)
            self.dialog.destroy()
            
        except ValueError:
            messagebox.showerror("Invalid Date", "Please enter dates in YYYY-MM-DD format")
    
    def _cancel(self):
        self.dialog.destroy()


def open_logs_window(parent: tk.Tk = None):
    """
    Open the logs viewing window
    
    Args:
        parent: Parent window
        
    Returns:
        LogsWindow instance
    """
    try:
        logs_window = LogsWindow(parent)
        logs_window.show_window()
        return logs_window
        
    except Exception as e:
        logger = get_module_logger("LogsWindow")
        logger.error(f"Failed to open logs window: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the logs window
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    try:
        logs_window = open_logs_window()
        root.mainloop()
    except Exception as e:
        print(f"Failed to run logs window: {str(e)}") 