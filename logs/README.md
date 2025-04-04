# Logs Directory  

This folder stores all the log files generated while preprocessing and training. We keep logs because our system resources are pretty limited, and training can take several hours. Most of the time, we let it run overnight, but it has crashed more times than we’d like to admit. Having logs helps us figure out when things went wrong, how long each run took, and where to pick up if something fails.  

### Why keep logs?  
- **Crashes happen** – If the system dies in the middle of the night, logs tell us where it left off.  
- **Tracking progress** – Knowing how long each step takes helps us optimize for future runs.  
- **Debugging & troubleshooting** – If something breaks, the logs usually have clues about why.  
- **Keeping a history** – Different runs might have different settings. Logs help us remember what worked and what didn’t.  

The log files in this directory are automatically generated during training and preprocessing. They’re ignored by version control, but we keep this README here so anyone working on this project knows why this folder exists.