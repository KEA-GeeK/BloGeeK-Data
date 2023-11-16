pipeline {
    agent any
    
    stages {
        stage('Check Main Branch Commit') {
            steps {
                script {
                    def branchName = env.BRANCH_NAME
                    def commitAuthor = sh(script: "git log --format='%an' -n 1", returnStdout: true).trim()
                    
                    if (branchName == 'main') {
                        currentBuild.result = 'FAILURE'
                        
                        // GitHub repository details
                        def githubOwner = 'KEA-GeeK' // GitHub owner/organization name
                        def githubRepo = 'BloGeeK-Data' // GitHub repository name
                        def githubToken = credentials('geek_data_test') // Jenkins credentials for GitHub token
                        
                        def githubAPIUrl = "https://api.github.com/repos/${githubOwner}/${githubRepo}/issues"
                        def issueTitle = 'Commits to main branch not allowed'
                        def issueBody = "Dear @${commitAuthor},\n\nPlease refrain from committing to the main branch.\n\nThank you!"
                        
                        def payload = [
                            title: issueTitle,
                            body: issueBody
                        ]
                        
                        def response = httpRequest(
                            httpMode: 'POST',
                            url: githubAPIUrl,
                            authentication: githubToken,
                            contentType: 'APPLICATION_JSON',
                            requestBody: groovy.json.JsonOutput.toJson(payload)
                        )
                        
                        if (response.status != 201) {
                            error "Failed to create GitHub issue: ${response.status} - ${response.content}"
                        } else {
                            echo "GitHub issue created successfully."
                        }
                    } else {
                        echo "Commit is not on the main branch. Skipping further action."
                    }
                }
            }
        }
    }
}
