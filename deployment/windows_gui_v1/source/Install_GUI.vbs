Option Explicit
Dim shell, fso, root, scriptPath, command, exitCode
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = root & "\Install_GUI.ps1"
shell.CurrentDirectory = root

MsgBox "Installation will run without a console window. It can take several minutes. Progress is recorded in logs\install.log.", vbInformation, "ConExtBiMamba GUI"
command = "powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & Chr(34) & scriptPath & Chr(34) & " -Runtime Auto"
exitCode = shell.Run(command, 0, True)

If exitCode = 0 Then
    MsgBox "Installation completed successfully. Double-click Launch_GUI.vbs to start the dashboard.", vbInformation, "ConExtBiMamba GUI"
Else
    MsgBox "Installation did not complete. Review logs\install.log. If GPU installation failed, try Install_GUI_CPU.vbs.", vbCritical, "ConExtBiMamba GUI"
End If
WScript.Quit exitCode

