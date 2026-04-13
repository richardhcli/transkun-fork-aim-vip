#====================================================================================================================================
#README: 
# This script serves as the epilogue for the whole pipeline of Transkun fine-tuning evaluations.
#====================================================================================================================================


#====================================================================================================================================
end_time=$(date +%s)

overall_runtime_seconds=$((end_time - START_TIME))
overall_runtime_minutes=$((overall_runtime_seconds / 60))
overall_runtime_formatted=$(printf "%02dh:%02dh:%02ds" $((overall_runtime_seconds / 3600)) $(( (overall_runtime_seconds % 3600) / 60)) $((overall_runtime_seconds % 60)))

#ending message:
MESSAGE="$RUNNING_SCRIPT completed at $(timestamp) in ${overall_runtime_formatted} seconds"

echo "[epilogue] $MESSAGE"

# Part,Description
# curl,The command-line tool used to send data over the web.
# -s,Silent mode. It hides the progress bar and error messages from your terminal.
# "-d ""...""","Data. This tells curl to send a POST request. The text inside the quotes is the actual message body, which uses your shell variables ($TARGET_ARCH and $overall_runtime_formatted) to tell you exactly what finished and how long it took."
# "-H ""Title: ...""",Header. Sets the title that appears on your phone's notification.
# "-H ""Topic: ...""",Header. A metadata tag used by ntfy to categorize or filter messages.
# ntfy.sh/cvirl...,"The Destination. This is the ""topic"" URL. Anyone (or any device) subscribed to ntfy.sh/cvirl-aortopathy will receive this message instantly."
# >/dev/null,"Redirect. This sends any output from the server into a ""black hole,"" ensuring the command doesn't print anything back to your terminal."

#notify me: 
RUNNING_SCRIPT_NAME=${RUNNING_SCRIPT_NAME:-${1:-RUNNING_SCRIPT}}
curl -s -d "$MESSAGE"  -H "Title: Training Finished: $RUNNING_SCRIPT_NAME" -H "Topic: purdue_vip_aim_richard_gilbrethjob" ntfy.sh/purdue_vip_aim_richard_gilbrethjob >/dev/null 
