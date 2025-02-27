tell application "Terminal"
    activate
    -- Set the project path
    set projectPath to "/Users/bryx/Documents/GitHub/Ward"
    
    -- Construct the command with error checking
    set command to "if [ -d '" & projectPath & "' ]; then"
    set command to command & " cd '" & projectPath & "';"
    set command to command & " if [ ! -x launch.sh ]; then chmod +x launch.sh; fi;"
    set command to command & " ./launch.sh;"
    set command to command & " else echo 'Error: Project directory not found!'; fi"
    
    -- Execute the command
    do script command
end tell
