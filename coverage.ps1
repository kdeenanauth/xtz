param($ignoreFlag)
Write-Host "Call 'coverage.ps1 disableopen' to avoid opening browser"
coverage run --source xtz setup.py test
coverage html
if ($PSBoundParameters["ignoreFlag"] -ne 'disableopen') {
    Invoke-Item .\htmlcov\index.html
}
