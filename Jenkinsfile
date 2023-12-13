pipeline {
    agent any
}

    stages {
        stage('Check Main Branch') {
            steps {
                script {
                    if (env.BRANCH_NAME != 'main') {
                        error("This pipeline only runs on the main branch.")
                    }
                }
            }
        }

        stage('Check Commit Message') {
            steps {
                script {
                    def commitMessage = sh(returnStdout: true, script: 'git log -1 --pretty=%B').trim()
                    def pattern = ~/(.+?) - .+/
                    def matcher = commitMessage =~ pattern

                    if (!matcher.matches()) {
                        error("Commit message does not follow the required format '(Model)-(Commit details)'.")
                    }

                    def modelFolder = matcher.group(1)
                    env.FOLDER_NAME = modelFolder
                    if (!fileExists(modelFolder)) {
                        error("Model folder '${modelFolder}' does not exist.")
                    }
                }
            }
        }

        stage('Build and Run Docker Image') {
            steps {
                script {
                    if (!fileExists('Dockerfile')) {
                        error("Dockerfile is missing.")
                    }

                    if (!fileExists('requirements.txt')) {
                        error("requirements.txt is missing.")
                    }

                    def imageName = "your_image:${env.GIT_COMMIT}"
                    sh "docker build -t ${imageName} ."
                    sh "docker run ${imageName}"
                }
            }
        }

        stage('Push to ECR') {
            steps {
                script {
                    def imageName = "your_image:${env.FOLDER_NAME}"
                    def ecrImageName = "${AWS_REGISTRY}/${imageName}"

                    sh "docker tag ${imageName} ${ecrImageName}"
                    sh "docker push ${ecrImageName}"
                }
            }
        }
    }

    post {
        failure {
            echo 'Pipeline failed.'
        }
    }
}
