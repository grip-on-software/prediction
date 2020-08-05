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
        string(name: 'PREDICTION_ORGANIZATIONS', defaultValue: "${env.ANALYSIS_ORGANIZATION}")
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
            archiveArtifacts artifacts: 'output/**/*.json', excludes: 'output/sprint_labels.json', onlyIfSuccessful: true
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
                    sh '/bin/bash -cex "rm -rf $PWD/output; mkdir $PWD/output; cd /home/docker; for org in $PREDICTION_ORGANIZATIONS; do Rscript features.r --core --log INFO --config $ANALYSIS_CONFIGURATION $REPORT_PARAMS --output $PWD/output --append --org \$org; done"'
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
                    sh "python files.py --upload output/sprint_features.arff --config $PREDICTOR_CONFIGURATION"
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
                sh "python tensor.py --filename output/sprint_features.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.json"
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
                }
            }
            steps {
                withCredentials([file(credentialsId: 'prediction-config', variable: 'PREDICTOR_CONFIGURATION')]) {
                    sh "python tensor.py --filename output/sprint_features.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.json --store owncloud --device /gpu:0 --config $PREDICTOR_CONFIGURATION"
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
                    sh "python files.py --download output/sprint_labels.json --remove output/sprint_features.arff output/sprint_labels.json --config $PREDICTOR_CONFIGURATION"
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
                    sh '/bin/bash -cex "cd /home/docker; for org in $PREDICTION_ORGANIZATIONS; do Rscript sprint_results.r --file $PWD/output/sprint_labels.json --features $PWD/output/sprint_features.arff --config $ANALYSIS_CONFIGURATION --output $PWD/output $REPORT_PARAMS --org $org; done"'
                }
            }
        }
    }
}
