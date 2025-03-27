import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import queue
import time

class LoadingScreen:
    def __init__(self, bootstyle="dark"):
        self.queue = queue.Queue()
        self.root = ttk.Toplevel()
        self.root.title("Loading...")
        self.root.geometry("400x250")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        
        self._center_window()
        
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)
        
        self.loading_label = ttk.Label(
            container,
            text="Initializing Application...",
            font=('Helvetica', 12),
            bootstyle=(bootstyle, "inverse")
        )
        self.loading_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            container,
            orient="horizontal",
            length=300,
            mode="determinate",  # Changed to determinate for better progress tracking
            bootstyle=(bootstyle, "striped")
        )
        self.progress.pack(pady=20)
        
        self.message_var = ttk.StringVar(value="Starting...")
        ttk.Label(
            container,
            textvariable=self.message_var,
            font=('Helvetica', 10),
            bootstyle=(bootstyle, "inverse")
        ).pack(pady=5)
        
        self.percent_var = ttk.StringVar(value="0%")
        ttk.Label(
            container,
            textvariable=self.percent_var,
            font=('Helvetica', 10, 'bold'),
            bootstyle=(bootstyle, "inverse")
        ).pack()
        
        # Start the queue checker before any updates occur
        self._active = True
        self._start_queue_checker()

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')

    def _start_queue_checker(self):
        def check_queue():
            try:
                while True:
                    msg = self.queue.get_nowait()
                    if msg == "complete":
                        self._handle_completion()
                        break
                    elif isinstance(msg, tuple):
                        self._handle_update(msg)
            except queue.Empty:
                if self._active:
                    self.root.after(50, check_queue)
        
        self.root.after(50, check_queue)

    def _handle_update(self, msg):
        message, *rest = msg
        self.message_var.set(message)
        if rest:
            percent = rest[0]
            self.progress['value'] = percent
            self.percent_var.set(f"{percent}%")
        self.root.update_idletasks()  # Force UI update

    def _handle_completion(self):
        self.progress['value'] = 100
        self.percent_var.set("100%")
        self.message_var.set("Ready!")
        self.root.update_idletasks()
        time.sleep(0.5)  # Brief pause so user sees completion
        self._active = False
        self.root.destroy()

    def update_status(self, message, percent=None):
        if percent is not None:
            self.queue.put((message, percent))
        else:
            self.queue.put((message,))

    def complete(self):
        self.queue.put("complete")

    def start_loading(self):
        self.root.grab_set()
        return self.root

def sturtap_loading_screen(initialization_func):
    loading_screen = LoadingScreen()
    
    # Start initialization in a separate thread
    init_thread = threading.Thread(target=initialization_func, args=(loading_screen.queue,))
    init_thread.daemon = True
    init_thread.start()
    
    # Run loading screen
    loading_screen.root.mainloop() 