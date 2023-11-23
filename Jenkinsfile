pipeline {
  agent any
  stages {
    stage('Check Main Branch Commit') {
      steps {
        script {
          pipeline {
            agent any

            stages {
              stage('Check if main branch') {
                steps {
                  script {
                    def currentBranch = sh(script: 'git rev-parse --abbrev-ref HEAD', returnStdout: true).trim()
                    if (currentBranch == 'main') {
                      currentBuild.result = 'FAILURE'
                      echo "This is the main branch. Sending notification through Github Issue."
                      // Implement GitHub issue notification here
                    } else {
                      echo "This is not the main branch."
                      currentBuild.result = 'SUCCESS'
                      checkout scm
                    }
                  }
                }
              }
            }
          }
        }

      }
    }

  }
}