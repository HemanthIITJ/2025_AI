### **File and Directory Management**
1. `Get-ChildItem` (alias: `ls`, `dir`) – Lists items in a directory.
2. `Set-Location` (alias: `cd`) – Changes the current directory.
3. `New-Item` – Creates a new file or directory.
4. `Remove-Item` (alias: `rm`, `del`) – Deletes files or directories.
5. `Copy-Item` (alias: `cp`) – Copies files or directories.
6. `Move-Item` (alias: `mv`) – Moves files or directories.
7. `Get-Content` (alias: `cat`, `type`) – Retrieves file contents.
8. `Set-Content` – Writes to a file.
9. `Add-Content` – Appends to a file.
10. `Clear-Content` – Clears file content.

### **System and Process Management**
11. `Get-Process` (alias: `ps`) – Lists running processes.
12. `Stop-Process` (alias: `kill`) – Stops a running process.
13. `Start-Process` – Starts a process.
14. `Get-Service` – Lists system services.
15. `Start-Service` – Starts a service.
16. `Stop-Service` – Stops a service.
17. `Restart-Service` – Restarts a service.
18. `Get-EventLog` – Retrieves event log entries.
19. `Get-Event` – Retrieves registered events.
20. `Clear-EventLog` – Clears event log entries.

### **System Information**
21. `Get-Host` – Shows PowerShell version and environment details.
22. `Get-ComputerInfo` – Retrieves detailed computer system information.
23. `Get-WmiObject` – Queries system data using WMI.
24. `Get-HotFix` – Shows installed updates or patches.
25. `Test-Connection` (alias: `ping`) – Tests network connectivity.
26. `Get-NetIPAddress` – Lists IP configuration details.

### **PowerShell Environment**
27. `Get-Command` – Lists available commands.
28. `Get-Help` (alias: `help`) – Shows help about commands.
29. `Import-Module` – Imports a module.
30. `Export-ModuleMember` – Exports module functions or variables.
31. `Get-Module` – Lists loaded modules.
32. `Remove-Module` – Unloads a module.

### **Variables and Data**
33. `Set-Variable` (alias: `sv`) – Defines a variable.
34. `Get-Variable` – Retrieves variable values.
35. `Remove-Variable` – Deletes variables.
36. `$Error` – A system variable storing error messages.
37. `$PSVersionTable` – Stores PowerShell version information.
38. `$null` – Represents an empty or null value.

### **Scripts and Automation**
39. `Invoke-Command` – Runs a script block remotely.
40. `Invoke-Expression` – Executes a string as a command.
41. `Read-Host` – Accepts user input.
42. `Write-Host` – Displays text output.
43. `Start-Sleep` (alias: `sleep`) – Pauses script execution.
44. `Set-ExecutionPolicy` – Modifies the script execution policy.
45. `ForEach-Object` (alias: `%`) – Loops through objects in a pipeline.
46. `Where-Object` (alias: `?`) – Filters objects based on conditions.

### **Security and Permissions**
47. `Get-Acl` – Retrieves access control lists (ACLs).
48. `Set-Acl` – Modifies ACLs.
49. `Get-Credential` – Prompts for user credentials.
50. `ConvertTo-SecureString` – Creates a secure string.
51. `New-Object` – Creates a .NET object instance.

### **Archives and Compression**
52. `Compress-Archive` – Creates ZIP files.
53. `Expand-Archive` – Extracts ZIP files.

### **Networking**
54. `Get-NetAdapter` – Lists network adapters.
55. `New-NetIPAddress` – Adds a new IP address.
56. `Get-DnsClient` – Shows DNS client configuration.
57. `Get-NetRoute` – Displays routing table entries.
58. `Invoke-WebRequest` (alias: `curl`, `wget`) – Fetches web content.
59. `Test-NetConnection` – Tests a network connection (ping, port check).

### **Date and Time**
60. `Get-Date` – Retrieves current date and time.
61. `Set-Date` – Modifies the system date and time.

### **Logging and Diagnostics**
62. `Out-File` – Writes output to a file.
63. `Out-Null` – Discards output.
64. `Start-Transcript` – Captures command output to a log file.
65. `Stop-Transcript` – Stops logging to a transcript.

### **Formatting and Output**
66. `Format-Table` (alias: `ft`) – Formats output as a table.
67. `Format-List` (alias: `fl`) – Formats output as a list.
68. `Sort-Object` (alias: `sort`) – Sorts objects.
69. `Group-Object` – Groups objects.
70. `Measure-Object` – Performs calculations like count, sum, average.

### **Package Management**
71. `Install-Package` – Installs software packages.
72. `Uninstall-Package` – Removes software packages.
73. `Find-Package` – Finds software packages.

### **Registry Management**
74. `Get-ItemProperty` – Reads registry key values.
75. `Set-ItemProperty` – Modifies registry key values.
76. `New-Item` – Creates a registry key.

### **Jobs and Background Tasks**
77. `Start-Job` – Runs a command in the background.
78. `Get-Job` – Retrieves background job status.
79. `Receive-Job` – Retrieves output from background jobs.
80. `Stop-Job` – Stops a background job.
81. `Remove-Job` – Deletes a background job.

### **Users and Groups**
82. `Get-LocalUser` – Lists local user accounts.
83. `New-LocalUser` – Creates a local user.
84. `Remove-LocalUser` – Deletes a local user.
85. `Get-LocalGroup` – Lists local groups.
86. `Add-LocalGroupMember` – Adds a user to a group.
87. `Remove-LocalGroupMember` – Removes a user from a group.

### **Special Commands**
88. `Switch` – Simplifies multi-condition branching.
89. `Try`, `Catch`, `Finally` – Error handling structure.
90. `Throw` – Generates a custom error.

### **Advanced Topics**
91. `Get-PSBreakpoint` – Manages breakpoints for debugging.
92. `Enable-PSBreakpoint` – Enables a breakpoint.
93. `Set-PSDebug` – Enables debugging.
94. `Compare-Object` – Compares two objects.
95. `Select-Object` (alias: `select`) – Filters object properties.
96. `New-PSSession` – Creates a persistent remote session.
97. `Enter-PSSession` – Enters a remote PowerShell session.
98. `Exit-PSSession` – Leaves the remote session.
99. `Export-Csv` – Exports objects to a CSV file.
100. `Import-Csv` – Imports data from a CSV file.



```markdown
# PowerShell File and Directory Management Commands

As an advanced PowerShell user, leveraging file and directory management commands efficiently can significantly enhance your scripting and automation tasks. Below is a detailed guide on these commands with practical examples tailored for senior developers.

## 1. Get-ChildItem (Aliases: `ls`, `dir`)
**Description:** Lists items (files and directories) in a specified location.

**Syntax:**
```powershell
Get-ChildItem -Path <string> [-Recurse] [-Filter <string>] [-Attributes <FileAttributes>]
```

**Examples:**
- **List all items in the current directory:**
  ```powershell
  Get-ChildItem
  ```
- **Recursively list all `.log` files modified in the last 7 days:**
  ```powershell
  Get-ChildItem -Path C:\Logs -Filter *.log -Recurse | Where-Object { $_.LastWriteTime -gt (Get-Date).AddDays(-7) }
  ```
- **List directories only:**
  ```powershell
  Get-ChildItem -Path . -Directory
  ```

---

## 2. Set-Location (Alias: `cd`)
**Description:** Changes the current working directory.

**Syntax:**
```powershell
Set-Location -Path <string>
```

**Examples:**
- **Change to the parent directory:**
  ```powershell
  Set-Location -Path ..
  ```
- **Navigate to the root of the C: drive:**
  ```powershell
  Set-Location -Path C:\
  ```
- **Change to a directory stored in a variable:**
  ```powershell
  $targetDir = 'C:\Projects\Current'
  Set-Location -Path $targetDir
  ```

---

## 3. New-Item
**Description:** Creates a new file or directory.

**Syntax:**
```powershell
New-Item -Path <string> -Name <string> -ItemType <File|Directory> [-Value <object>]
```

**Examples:**
- **Create a new directory named `Archive`:**
  ```powershell
  New-Item -Path . -Name 'Archive' -ItemType 'Directory'
  ```
- **Create a new script file with initial content:**
  ```powershell
  $scriptContent = @"
  Param(\$param1, \$param2)
  # Script logic goes here
  "@
  New-Item -Path . -Name 'Deploy.ps1' -ItemType 'File' -Value $scriptContent
  ```
- **Create multiple files in a loop:**
  ```powershell
  1..5 | ForEach-Object { New-Item -Path . -Name "File$_.txt" -ItemType 'File' }
  ```

---

## 4. Remove-Item (Aliases: `rm`, `del`)
**Description:** Deletes files or directories.

**Syntax:**
```powershell
Remove-Item -Path <string> [-Recurse] [-Force] [-Confirm]
```

**Examples:**
- **Delete a single file with confirmation:**
  ```powershell
  Remove-Item -Path .\debug.log -Confirm
  ```
- **Forcefully remove a directory and all its contents:**
  ```powershell
  Remove-Item -Path C:\Temp\OldBuilds -Recurse -Force
  ```
- **Delete all `.tmp` files in a directory:**
  ```powershell
  Get-ChildItem -Path C:\Temp -Filter *.tmp -File | Remove-Item
  ```

---

## 5. Copy-Item (Alias: `cp`)
**Description:** Copies files or directories from one location to another.

**Syntax:**
```powershell
Copy-Item -Path <string> -Destination <string> [-Recurse] [-Force]
```

**Examples:**
- **Copy a file to a backup location:**
  ```powershell
  Copy-Item -Path .\database.db -Destination C:\Backup\database.db
  ```
- **Recursively copy a directory to another drive:**
  ```powershell
  Copy-Item -Path C:\Projects\MyApp -Destination D:\ProjectsBackup\MyApp -Recurse
  ```
- **Copy files matching a pattern:**
  ```powershell
  Copy-Item -Path .\*.config -Destination .\ConfigBackup\
  ```

---

## 6. Move-Item (Alias: `mv`)
**Description:** Moves files or directories to a new location or renames them.

**Syntax:**
```powershell
Move-Item -Path <string> -Destination <string> [-Force]
```

**Examples:**
- **Move log files to an archive folder:**
  ```powershell
  Move-Item -Path C:\Logs\*.log -Destination C:\Logs\Archive\
  ```
- **Rename a file:**
  ```powershell
  Move-Item -Path .\README.txt -Destination .\README_old.txt
  ```
- **Move and overwrite existing files:**
  ```powershell
  Move-Item -Path C:\Source\* -Destination C:\Destination\ -Force
  ```

---

## 7. Get-Content (Aliases: `cat`, `type`)
**Description:** Retrieves the contents of a file.

**Syntax:**
```powershell
Get-Content -Path <string> [-TotalCount <int>] [-Tail <int>] [-Wait]
```

**Examples:**
- **Read the contents of a configuration file:**
  ```powershell
  Get-Content -Path .\appsettings.json
  ```
- **Tail a log file in real-time:**
  ```powershell
  Get-Content -Path .\access.log -Wait
  ```
- **Read environment-specific settings:**
  ```powershell
  $env = 'Production'
  Get-Content -Path ".\config.$env.xml"
  ```

---

## 8. Set-Content
**Description:** Writes content to a file, overwriting existing content.

**Syntax:**
```powershell
Set-Content -Path <string> -Value <object> [-Encoding <string>]
```

**Examples:**
- **Overwrite a file with new content:**
  ```powershell
  $jsonContent = @{ Name = 'App'; Version = '2.0' } | ConvertTo-Json
  Set-Content -Path .\appsettings.json -Value $jsonContent
  ```
- **Set content with specific encoding:**
  ```powershell
  Set-Content -Path .\script.ps1 -Value $scriptContent -Encoding UTF8
  ```

---

## 9. Add-Content
**Description:** Appends content to the end of a file.

**Syntax:**
```powershell
Add-Content -Path <string> -Value <object>
```

**Examples:**
- **Append a line to a text file:**
  ```powershell
  Add-Content -Path .\notes.txt -Value "Don't forget to review the PRs."
  ```
- **Log events with timestamps:**
  ```powershell
  $event = "User login at $(Get-Date)"
  Add-Content -Path .\event.log -Value $event
  ```

---

## 10. Clear-Content
**Description:** Clears the content of a file without deleting the file itself.

**Syntax:**
```powershell
Clear-Content -Path <string>
```

**Examples:**
- **Clear the contents of a log file:**
  ```powershell
  Clear-Content -Path .\session.log
  ```

---

## Additional Tips for Advanced Usage
1. **Using `-WhatIf` and `-Confirm`:**
   Simulate commands before execution:
   ```powershell
   Remove-Item -Path .\important\* -Recurse -WhatIf
   ```

2. **Error Handling with Try-Catch:**
   ```powershell
   try {
       Move-Item -Path .\data.csv -Destination .\backup\data.csv -ErrorAction Stop
   } catch {
       Write-Error "Failed to move file: $_"
   }
   ```




#### 11. Get-Process (Alias: `ps`)
**Description:** Retrieves information about the processes running on a local or remote computer.

**Syntax:**
```powershell
Get-Process [[-Name] <string[]>] [-ComputerName <string[]>] [-FileVersionInfo] [-Module] [-IncludeUserName]
```

**Examples:**
- **List all running processes:**
  ```powershell
  Get-Process
  ```
- **Get processes by name (e.g., all instances of Notepad):**
  ```powershell
  Get-Process -Name notepad
  ```
- **Display detailed information including the user running each process:**
  ```powershell
  Get-Process -IncludeUserName
  ```
- **Sort processes by memory usage in descending order:**
  ```powershell
  Get-Process | Sort-Object -Property WS -Descending
  ```
- **Get processes running on a remote computer:**
  ```powershell
  Get-Process -ComputerName Server01
  ```
  **Note:** Ensure that remote management is enabled and you have the necessary permissions.

---

#### 12. Stop-Process (Alias: `kill`)
**Description:** Stops one or more running processes.

**Syntax:**
```powershell
Stop-Process [-Id] <int[]> [-Force] [-PassThru]
Stop-Process [-Name] <string[]> [-Force] [-PassThru]
```

**Examples:**
- **Stop a process by its ID:**
  ```powershell
  Stop-Process -Id 1234
  ```
- **Forcefully stop multiple processes by name:**
  ```powershell
  Stop-Process -Name chrome, notepad -Force
  ```
- **Stop all instances of a process and output the stopped processes:**
  ```powershell
  Stop-Process -Name 'iexplore' -PassThru
  ```
- **Stop processes consuming more than 500MB of memory:**
  ```powershell
  Get-Process | Where-Object { $_.WS -gt 500MB } | Stop-Process
  ```
  **Caution:** Ensure you do not inadvertently stop critical system processes.

---

#### 13. Start-Process
**Description:** Starts one or more processes on the local computer.

**Syntax:**
```powershell
Start-Process [-FilePath] <string> [-ArgumentList <string[]>] [-WorkingDirectory <string>] [-Verb <string>] [-Wait]
```

**Examples:**
- **Start Notepad:**
  ```powershell
  Start-Process -FilePath notepad.exe
  ```
- **Open a URL in the default browser:**
  ```powershell
  Start-Process -FilePath 'http://www.example.com'
  ```
- **Run a script with arguments and wait for it to complete:**
  ```powershell
  Start-Process -FilePath PowerShell.exe -ArgumentList '-File', '.\Deploy.ps1', '-Environment', 'Prod' -Wait
  ```
- **Start a process with elevated privileges:**
  ```powershell
  Start-Process -FilePath notepad.exe -Verb RunAs
  ```
  **Note:** This will prompt for administrative credentials.

---

#### 14. Get-Service
**Description:** Gets the services on a local or remote computer.

**Syntax:**
```powershell
Get-Service [[-Name] <string[]>] [-ComputerName <string[]>]
```

**Examples:**
- **List all services:**
  ```powershell
  Get-Service
  ```
- **Get services with names starting with 'SQL':**
  ```powershell
  Get-Service -Name 'SQL*'
  ```
- **Check the status of a specific service:**
  ```powershell
  Get-Service -Name W32Time
  ```
- **Get services on a remote computer:**
  ```powershell
  Get-Service -ComputerName Server01
  ```

---

#### 15. Start-Service
**Description:** Starts a stopped service.

**Syntax:**
```powershell
Start-Service [-Name] <string[]> [-PassThru] [-Verbose]
```

**Examples:**
- **Start a single service:**
  ```powershell
  Start-Service -Name W32Time
  ```
- **Start multiple services and display their status:**
  ```powershell
  Start-Service -Name 'Spooler', 'W32Time' -PassThru
  ```
- **Start all services that are set to start automatically but are currently stopped:**
  ```powershell
  Get-Service | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' } | Start-Service
  ```

---

#### 16. Stop-Service
**Description:** Stops a running service.

**Syntax:**
```powershell
Stop-Service [-Name] <string[]> [-Force] [-PassThru] [-Verbose]
```

**Examples:**
- **Stop a service gracefully:**
  ```powershell
  Stop-Service -Name 'Spooler'
  ```
- **Forcefully stop a service:**
  ```powershell
  Stop-Service -Name 'Spooler' -Force
  ```
- **Stop multiple services and output their final status:**
  ```powershell
  Stop-Service -Name 'Spooler', 'W32Time' -PassThru
  ```
- **Stop services consuming high CPU usage:**
  ```powershell
  Get-Service | Where-Object { $_.CPU -gt 80 } | Stop-Service
  ```
  **Note:** The `CPU` property may not be directly available; you might need to correlate with processes.

---

#### 17. Restart-Service
**Description:** Stops and then starts one or more services.

**Syntax:**
```powershell
Restart-Service [-Name] <string[]> [-Force] [-PassThru] [-Verbose]
```

**Examples:**
- **Restart a service:**
  ```powershell
  Restart-Service -Name 'Spooler'
  ```
- **Forcefully restart a service:**
  ```powershell
  Restart-Service -Name 'WSearch' -Force
  ```
- **Restart multiple services and output their status:**
  ```powershell
  Restart-Service -Name 'W32Time', 'Dnscache' -PassThru
  ```
- **Restart services that have been running for more than 7 days:**
  ```powershell
  Get-Service | Where-Object { (Get-Date) - $_.ServicesDependedOn.StartTime -gt [TimeSpan]::FromDays(7) } | Restart-Service
  ```
  **Note:** `ServicesDependedOn.StartTime` may not be directly accessible; additional methods may be needed.

---

#### 18. Get-EventLog
**Description:** Gets events and event logs from local and remote computers.

**Syntax:**
```powershell
Get-EventLog [-LogName] <string> [[-InstanceId] <int64[]>] [-Newest <int32>] [-EntryType <string[]>]
```

**Examples:**
- **List all event logs on the system:**
  ```powershell
  Get-EventLog -List
  ```
- **Get the last 50 entries from the System log:**
  ```powershell
  Get-EventLog -LogName 'System' -Newest 50
  ```
- **Filter events by event ID:**
  ```powershell
  Get-EventLog -LogName 'Application' -InstanceId 1000
  ```
- **Get error entries from the Application log:**
  ```powershell
  Get-EventLog -LogName 'Application' -EntryType 'Error'
  ```
- **Retrieve events from a remote computer:**
  ```powershell
  Get-EventLog -LogName 'System' -Newest 100 -ComputerName Server01
  ```

---

#### 19. Get-Event
**Description:** Gets events in the PowerShell event queue.

**Syntax:**
```powershell
Get-Event [[-SourceIdentifier] <string[]>] [-Force]
```

**Examples:**
- **List all events in the event queue:**
  ```powershell
  Get-Event
  ```
- **Get events by source identifier:**
  ```powershell
  Get-Event -SourceIdentifier 'ProcessStarted'
  ```
- **Wait for a specific event (e.g., a process exit):**
  ```powershell
  $process = Start-Process -FilePath 'notepad.exe' -PassThru
  Register-ObjectEvent -InputObject $process -EventName 'Exited' -SourceIdentifier 'NotepadExited'
  Wait-Event -SourceIdentifier 'NotepadExited'
  Write-Host 'Notepad has exited.'
  ```
- **Remove event subscriptions:**
  ```powershell
  Unregister-Event -SourceIdentifier 'NotepadExited'
  ```

---

#### 20. Clear-EventLog
**Description:** Deletes all entries from specified event logs on the local or remote computers.

**Syntax:**
```powershell
Clear-EventLog [-LogName] <string[]> [-ComputerName <string[]>]
```

**Examples:**
- **Clear the Application event log:**
  ```powershell
  Clear-EventLog -LogName 'Application'
  ```
- **Clear multiple event logs:**
  ```powershell
  Clear-EventLog -LogName 'System', 'Security'
  ```
- **Clear event logs on a remote computer:**
  ```powershell
  Clear-EventLog -

