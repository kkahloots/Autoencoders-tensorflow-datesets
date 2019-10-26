
import os
import tensorflow as tf

class Logger:
    def __init__(self, sess, log_dir):
        self.sess = sess
        self.log_dir = log_dir
        self.log_placeholders = {}
        self.log_ops = {}
        self.train_log_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"),
                                                          self.sess.graph)
        self.test_log_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", log_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param log_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_log_writer if summarizer == "train" else self.test_log_writer
        with tf.variable_scope(scope):

            if log_dict is not None:
                log_list = []
                for tag, value in log_dict.items():
                    if tag not in self.log_ops:
                        if len(value.shape) <= 1:
                            self.log_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.log_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.log_ops[tag] = tf.summary.scalar(tag, self.log_placeholders[tag])
                        else:
                            self.log_ops[tag] = tf.summary.image(tag, self.log_placeholders[tag])

                    log_list.append(self.sess.run(self.log_ops[tag], {self.log_placeholders[tag]: value}))

                for log in log_list:
                    summary_writer.add_summary(log, step)
                summary_writer.flush()
                
