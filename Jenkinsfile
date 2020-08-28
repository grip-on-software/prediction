pipeline {
    agent { label 'docker' }

    environment {
        COLLECTOR_IMAGE = "gros-data-analysis-dashboard"
        PREDICTOR_TAG = env.BRANCH_NAME.replaceFirst('^master$', 'latest')
        PREDICTOR_NAME = "gros-prediction"
        PREDICTOR_IMAGE = "${env.PREDICTOR_NAME}:${env.PREDICTOR_TAG}"
        GITLAB_TOKEN = credentials('prediction-gitlab-token')
    }

    parameters {
        string(name: 'PREDICTION_ARGS', defaultValue: '--label num_not_done_points+num_removed_points+num_added_points --binary --roll-sprints --roll-validation --roll-labels --replace-na --model dnn --test-interval 200 --num-epochs 1000', description: 'Prediction arguments')
        string(name: 'PREDICTION_ORGANIZATIONS', defaultValue: "${env.PREDICTION_ORGANIZATIONS}", description: 'Organizations to include in prediction')
    }
    options {
        gitLabConnection('gitlab')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    triggers {
        gitlab(triggerOnPush: true, triggerOnMergeRequest: true, branchFilterType: 'All', secretToken: env.GITLAB_TOKEN)
        cron(env.PREDICTION_CRON)
    }

    post {
        success {
            archiveArtifacts artifacts: 'output/**/*.json', excludes: 'output/sprint_labels*.json', onlyIfSuccessful: true
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'success'
        }
        failure {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'failed'
        }
        aborted {
            updateGitlabCommitStatus name: env.JOB_NAME, state: 'canceled'
        }
    }

    stages {
        stage('Build') {
            steps {
                checkout scm
                sh 'docker build -t $DOCKER_REPOSITORY/$PREDICTOR_IMAGE . --build-arg TENSORFLOW_VERSION=$TENSORFLOW_VERSION'
            }
        }
        stage('Push') {
            steps {
                withDockerRegistry(credentialsId: 'docker-credentials', url: env.DOCKER_URL) {
                    sh 'docker push $DOCKER_REPOSITORY/$PREDICTOR_IMAGE'
                }
            }
        }
        stage('Collect') {
            agent {
                docker {
                    image '$COLLECTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh "/bin/bash -cex \"rm -rf \$WORKSPACE/output; mkdir \$WORKSPACE/output; cd /home/docker; for org in ${params.PREDICTION_ORGANIZATIONS}; do Rscript features.r --core --log INFO --config $ANALYSIS_CONFIGURATION $REPORT_PARAMS --output \$WORKSPACE/output --filename sprint_features.${env.BUILD_TAG}.arff --append --org \\\$org; done\""
                }
            }
        }
        stage('Upload') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image '$PREDICTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '-v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'master'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python files.py --upload output/sprint_features.${env.BUILD_TAG}.arff --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Predict') {
            when {
                beforeAgent true
                not { environment name: 'PREDICTOR_REMOTE', value: 'true' }
            }
            agent {
                docker {
                    image '$PREDICTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                sh "python tensor.py --filename output/sprint_features.${env.BUILD_TAG}.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.json"
            }
        }
        stage('Predict remote GPU') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image '$PREDICTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '--runtime=nvidia -v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'gpu'
                    alwaysPull true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python tensor.py --filename output/sprint_features.${env.BUILD_TAG}.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.${env.BUILD_TAG}.json --store owncloud --device /gpu:${env.EXECUTOR_NUMBER} --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Download') {
            when {
                beforeAgent true
                environment name: 'PREDICTOR_REMOTE', value: 'true'
            }
            agent {
                docker {
                    image '$PREDICTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    args '-v /usr/local/share/ca-certificates/:/usr/local/share/ca-certificates/'
                    label 'master'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python files.py --download output/sprint_labels.${env.BUILD_TAG}.json --remove output/sprint_features.${env.BUILD_TAG}.arff output/sprint_labels.${env.BUILD_TAG}.json --config $PREDICTOR_CONFIGURATION"
                }
            }
        }
        stage('Format') {
            agent {
                docker {
                    image '$COLLECTOR_IMAGE'
                    registryUrl "${env.DOCKER_URL}"
                    registryCredentialsId 'docker-credentials'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh "/bin/bash -cex \"cd /home/docker; for org in ${params.PREDICTION_ORGANIZATIONS}; do Rscript sprint_results.r --file \$WORKSPACE/output/sprint_labels.${env.BUILD_TAG}.json --features \$WORKSPACE/output/sprint_features.${env.BUILD_TAG}.arff --config $ANALYSIS_CONFIGURATION --output \$WORKSPACE/output $REPORT_PARAMS --org \\\$org; done\""
                }
            }
        }
    }
}
