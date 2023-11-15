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
                        // 여기에 마스터 브랜치에서 실행할 작업 추가
                    } else if (currentBranch == 'develop') {
                        echo 'Building on the develop branch...'
                        // 여기에 develop 브랜치에서 실행할 작업 추가
                    } else {
                        echo 'Building on a different branch...'
                        // 다른 브랜치에서 실행할 작업 추가
                    }
                }
            }
        }
        // 여기에 다른 스테이지들을 추가하여 작업을 정의할 수 있습니다.
    }
    // 파이프라인의 다른 설정들을 추가할 수 있습니다.
}
