pipeline {
  agent any
  stages {
    stage('Check Branch') {
      steps {
        script {
          def currentBranch = env.BRANCH_NAME
          echo "Currently building on branch: ${currentBranch}"

          if (currentBranch == 'master') {

            echo 'Building on the master branch...'

          } else if (currentBranch == 'develop') {

            echo 'Building on the develop branch...'

          } else {

            echo 'Building on a different branch...'

          }
        }

      }
    }

  }
}