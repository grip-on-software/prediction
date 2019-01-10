pipeline {
    agent { label 'docker' }

    environment {
        COLLECTOR_IMAGE = "${env.DOCKER_REGISTRY}/gros-data-analysis-dashboard"
        PREDICTOR_TAG = env.BRANCH_NAME.replaceFirst('^master$', 'latest')
        PREDICTOR_NAME = "${env.DOCKER_REGISTRY}/gros-prediction"
        PREDICTOR_IMAGE = "${env.PREDICTOR_NAME}:${env.PREDICTOR_TAG}"
        GITLAB_TOKEN = credentials('prediction-gitlab-token')
    }

    parameters {
        string(name: 'PREDICTION_ARGS', defaultValue: '--label num_not_done_points+num_removed_points+num_added_points --binary --roll-sprints --roll-validation --roll-labels --replace-na --model dnn --test-interval 200 --num-epochs 1000', description: 'Prediction arguments')
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
                sh 'docker build -t $PREDICTOR_IMAGE . --build-arg TENSORFLOW_VERSION=$TENSORFLOW_VERSION'
            }
        }
        stage('Push') {
            steps {
                sh 'docker push $PREDICTOR_IMAGE'
            }
        }
        stage('Collect') {
            agent {
                docker {
                    image '$COLLECTOR_IMAGE'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh '/bin/bash -c "rm -rf $PWD/output && mkdir $PWD/output && cd /home/docker && Rscript features.r --core --log INFO --config $ANALYSIS_CONFIGURATION --output $PWD/output $REPORT_PARAMS"'
                }
            }
        }
        stage('Predict') {
            agent {
                docker {
                    image '$PREDICTOR_IMAGE'
                    reuseNode true
                }
            }
            steps {
                sh "python tensor.py --filename output/sprint_features.arff --log INFO --seed 123 --clean ${params.PREDICTION_ARGS} --results output/sprint_labels.json"
            }
        }
        stage('Format') {
            agent {
                docker {
                    image '$COLLECTOR_IMAGE'
                    reuseNode true
                }
            }
            steps {
                withCredentials([file(credentialsId: 'data-analysis-config', variable: 'ANALYSIS_CONFIGURATION')]) {
                    sh '/bin/bash -c "cd /home/docker && Rscript sprint_results.r --file $PWD/output/sprint_labels.json --features $PWD/output/sprint_features.arff --config $ANALYSIS_CONFIGURATION --output $PWD/output $REPORT_PARAMS"'
                }
            }
        }
    }
}
