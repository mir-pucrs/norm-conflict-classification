#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script deals with the configuration file ``settings.conf``
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from os.path import realpath, abspath

class Configuration(object):

    def __init__(self, configfile=None):
        self.fconf = configfile
        self.dic = {}

        if configfile:
            self.input = open(configfile)
        else:
            self.input = open(abspath('../settings.conf'))
        self._loadcontent()

    
    def __str__(self):
        """ Return the content of the class """
        content = 'Variables: \n'
        for k in self.dic:
            content += '%s: $s\n'+self.dic[k]
        return content

    
    def _loadcontent(self):
        for line in self.input:
            line = line.strip()
            if not line and line.startswith('#'):
                continue

            # General content
            if '=' in line:
                var, value = line.split('=')
                for k in self.dic:
                    if '$'+k in value:
                        value = value.replace('$'+k, self.dic[k])
                self.dic[var] = value


    def get(self, key):
        if self.dic.has_key(key):
            value = self.dic[key]
            if value.isdigit():
                return int(value)
            elif value == 'False':
                return False
            elif value == 'True':
                return True
            return value
        else:
            return ''


    def exists(self, key, value, error=True):
        if self.dic.has_key(key):
            arr = self.dic[key].split(',')
            if value in arr:
                return True
        if error:
            logger.error('The value %s does not exists in %s' % (value, key))
            sys.exit(0)
        return False


    def close(self):
        if self.input:
            self.input.close()
# End of class Configuration
