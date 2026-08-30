Option Explicit
Dim shell, fso, root, scriptPath, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = root & "\Install_GUI.ps1"
shell.CurrentDirectory = root

MsgBox "CPU-only installation will run without a console window. Progress is recorded in logs\install.log.", vbInformation, "ConExtBiMamba GUI"
command = "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & Chr(34) & scriptPath & Chr(34) & " -Runtime CPU"
exitCode = shell.Run(command, 0, True)

If exitCode = 0 Then
    MsgBox "CPU-only installation completed successfully. Double-click Launch_GUI.vbs to start the dashboard.", vbInformation, "ConExtBiMamba GUI"
Else
    MsgBox "Installation did not complete. Review logs\install.log for the exact error.", vbCritical, "ConExtBiMamba GUI"
End If
WScript.Quit exitCode

