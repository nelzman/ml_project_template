library(azuremlsdk)  # sdk 1, deprecated 2021: https://azure.github.io/azureml-sdk-for-r/

# initial setup:
# 1. Resource Group 
# 

# Set the workspace details
subscription_id <- "<your_subscription_id>"
resource_group <- "<your_resource_group>"
workspace_name <- "<your_workspace_name>"

new_ws <- create_workspace(name = workspace_name, 
  subscription_id = subscription_id, 
  resource_group = resource_group_name, 
  location = location, 
  create_resource_group = FALSE
)

# Authenticate and connect to the workspace
ws <- get_workspace(
    subscription_id = subscription_id,
    resource_group = resource_group,
    workspace_name = workspace_name
)

# Verify the connection
print(ws)

